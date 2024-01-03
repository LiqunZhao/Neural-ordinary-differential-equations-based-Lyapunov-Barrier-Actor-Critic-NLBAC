""" This environment is adapted from:
https://github.com/yemam3/Mod-RL-RCBF

"""
import numpy as np
import gym
from gym import spaces
from envs.utils import to_pixel

class UnicycleEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self,seed):

        super(UnicycleEnv, self).__init__()

        self.dynamics_mode = 'Unicycle'
        # Define action and observation space
        # They must be gym.spaces objects
        low_safe_bound = np.array([-3.5,-12])
        high_safe_bound = np.array([3.5,12])
        self.safe_action_space = spaces.Box(low=low_safe_bound, high=high_safe_bound)
        self.action_space = spaces.Box(low=low_safe_bound, high=high_safe_bound)
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(7,))
        self.bds = np.array([[-3., -3.], [3., 3.]])
        self.hazards_radius = 0.5
        self.hazards_locations = np.array([[0., 0.], [0.,1.],[0.,-1.],[-1., 1.], [-1., -1.], [1., -1.], [1., 1.]]) * 1.5
        np.random.seed(seed)

        self.dt = 0.02
        self.max_episode_steps = 1200
        self.reward_goal = 500
        self.goal_size = 0.3
        # Initialize Env
        self.state = None
        self.episode_step = 0
        self.goal_pos = np.array([2.5, 2.5])
        self.center = np.array([-2.47,-2.5])           #used for lyapunov, the center position before the step function
        self.next_center = np.array([-2.47,-2.5])      #used for updating the lyapunov function, the center position after te step function

        self.reset()
        # Get Dynamics
        self.get_f, self.get_g = self._get_dynamics()
        # Disturbance
        self.disturb_mean = np.zeros((3,))
        self.disturb_covar = np.diag([0.005, 0.005, 0.05]) * 20
        self.safety_cost_coef = 1.0

        # Viewer
        self.viewer = None

    def step(self, action):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        action : ndarray
                Action that the agent takes in the environment

        Returns
        -------
        See the returns of the function _step()

        """
        state, reward, constraint,center_pos,next_center_pos, done, info = self._step(action)
        return self.get_obs(), reward, constraint,center_pos,next_center_pos, done, info

    def _step(self, action):
        """

        Parameters
        ----------
        action

        Returns
        -------
        state : ndarray
            New internal state of the agent.
        reward : float
            Reward collected during this transition.
        constraint: float
            Constraint collected as cost for approximating the Lyapunov function
        center_pos: ndarray
            Position before the action is taken. This will be used as the input of the Lyapunov network.
        next_center_pos: ndarray
            Position after the action is taken. This will be used as the input of the Lyapunov network.
        done : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """

        l_p = 0.03
        x0 = self.state[0] + l_p * np.cos(self.state[2])
        x1 = self.state[1] + l_p * np.sin(self.state[2])
        self.center = np.array([x0, x1])
        center_pos = self.center                #current center position before stepping the agent, used to train the lyapunov


        des_v = 2.5                             # a pre-defined velocity, want the agent to keep this velocity
        self.state += self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action)
        self.state -= self.dt * 0.1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]),  0])
        l_p = 0.03
        x0 = self.state[0] + l_p * np.cos(self.state[2])
        x1 = self.state[1] + l_p * np.sin(self.state[2])
        self.next_center = np.array([x0, x1])
        next_center_pos = self.next_center  #center position after stepping the agent, it is used for constructing the lyapunov_target to update the lyapunov network used as CLF.

        self.episode_step += 1

        info = dict()

        dist_goal = self._goal_dist_center()
        constraint = dist_goal
        velocity = action[0]
        reward_velocity = -np.square(velocity - des_v) * 0.1    # one part of the reward, which means that we want the velocity can be kept as close as possible to the v_desired
        reward_forward = (self.last_goal_dist - dist_goal) * 30
        reward = reward_velocity + reward_forward
        self.last_goal_dist = dist_goal
        # Check if goal is met
        if self.goal_met():
            info['goal_met'] = True
            reward += self.reward_goal
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps


        # Include constraint cost in info
        if np.any(np.sum((self.next_center - self.hazards_locations)**2, axis=1) < self.hazards_radius**2):
            num_cbfs = self.hazards_locations.shape[0]
            for i in range(num_cbfs):
                if np.sum((self.next_center - self.hazards_locations[i])**2, axis=0) < self.hazards_radius**2:


                    if 'num_safety_violation' in info:
                        info['num_safety_violation'] += 1
                    else:
                        info['num_safety_violation'] = 1


                    if 'safety_cost' in info:
                        dist_to_center = np.sqrt(np.sum((self.next_center - self.hazards_locations[i])**2, axis=0))
                        safety_cost_val = ((self.hazards_radius - dist_to_center) / self.hazards_radius) * self.safety_cost_coef
                        info['safety_cost'] += safety_cost_val
                    else:
                        dist_to_center = np.sqrt(np.sum((self.next_center - self.hazards_locations[i]) ** 2, axis=0))
                        safety_cost_val = ((self.hazards_radius - dist_to_center) / self.hazards_radius) * self.safety_cost_coef
                        info['safety_cost'] = safety_cost_val                                                           # Used by other algorithms like CPO, PPO-Lag and TRPO-Lag

        return self.state, reward,constraint,center_pos,next_center_pos, done, info

    def goal_met(self):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """

        return np.linalg.norm(self.next_center - self.goal_pos) <= self.goal_size

    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.episode_step = 0

        # Re-initialize state
        self.state = np.array([-2.5, -2.5, 0.])
        self.center = np.array([-2.47, -2.5])  # used for lyapunov, the center position before the step function
        self.next_center = np.array([-2.47, -2.5])  # used for updating the lyapunov function, the center position after te step function

        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist_center()

        return self.get_obs()

    def render(self, mode='human', close=False):
        """Render the environment to the screen

         Parameters
         ----------
         mode : str
         close : bool

         Returns
         -------

         """

        if mode != 'human' and mode != 'rgb_array':
            rel_loc = self.goal_pos - self.state[:2]
            theta_error = np.arctan2(rel_loc[1], rel_loc[0]) - self.state[2]
            print('Ep_step = {}, \tState = {}, \tDist2Goal = {}, alignment_error = {}'.format(self.episode_step,
                                                                                              self.state,
                                                                                              self._goal_dist_center(),
                                                                                              theta_error))

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from envs import pyglet_rendering
            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            # Draw obstacles
            obstacles = []
            for i in range(len(self.hazards_locations)):
                obstacles.append(
                    pyglet_rendering.make_circle(radius=to_pixel(self.hazards_radius, shift=0), filled=True))
                obs_trans = pyglet_rendering.Transform(translation=(
                to_pixel(self.hazards_locations[i][0], shift=screen_width / 2),
                to_pixel(self.hazards_locations[i][1], shift=screen_height / 2)))
                obstacles[i].set_color(1.0, 0.0, 0.0)
                obstacles[i].add_attr(obs_trans)
                self.viewer.add_geom(obstacles[i])

            # Make Goal
            goal = pyglet_rendering.make_circle(radius=to_pixel(self.goal_size, shift=0), filled=True)
            goal_trans = pyglet_rendering.Transform(translation=(
            to_pixel(self.goal_pos[0], shift=screen_width / 2), to_pixel(self.goal_pos[1], shift=screen_height / 2)))
            goal.add_attr(goal_trans)
            goal.set_color(0.0, 0.5, 0.0)
            self.viewer.add_geom(goal)

            # Make Robot
            self.robot = pyglet_rendering.make_circle(radius=to_pixel(0.1), filled=True)
            self.robot_trans = pyglet_rendering.Transform(translation=(
            to_pixel(self.state[0], shift=screen_width / 2), to_pixel(self.state[1], shift=screen_height / 2)))
            self.robot_trans.set_rotation(self.state[2])
            self.robot.add_attr(self.robot_trans)
            self.robot.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.robot)
            self.robot_orientation = pyglet_rendering.Line(start=(0.0, 0.0), end=(15.0, 0.0))
            self.robot_orientation.linewidth.stroke = 2
            self.robot_orientation.add_attr(self.robot_trans)
            self.robot_orientation.set_color(0, 0, 0)
            self.viewer.add_geom(self.robot_orientation)

        if self.state is None:
            return None

        self.robot_trans.set_translation(to_pixel(self.state[0], shift=screen_width / 2),
                                         to_pixel(self.state[1], shift=screen_height / 2))
        self.robot_trans.set_rotation(self.state[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
        """

        rel_loc = self.goal_pos - self.state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()  # compass to the goal

        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), goal_compass[0], goal_compass[1], np.exp(-goal_dist)])

    def _get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        def get_f(state):
            f_x = np.zeros(state.shape)
            return f_x

        def get_g(state):
            theta = state[2]
            g_x = np.array([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [            0, 1.0]])
            return g_x

        return get_f, get_g

    def obs_compass(self):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

        # Get ego vector in world frame
        vec = self.goal_pos - self.state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _goal_dist_center(self):
        return np.linalg.norm(self.goal_pos - self.next_center)     #use self.next_center instead of self.center since we need to use St+1 for c(st,at)

