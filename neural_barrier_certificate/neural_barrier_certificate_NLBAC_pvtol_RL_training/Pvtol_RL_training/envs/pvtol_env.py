import numpy as np
import gym
from gym import spaces
from envs.utils import to_pixel
# from Pvtol_RL_training.sac_cbf_clf.utils import get_polygon_normals, prYellow, prRed

class PvtolEnv(gym.Env):
    """Custom Environment that follows SafetyGym interface"""

    metadata = {'render.modes': ['human']}

    def __init__(self,seed):

        super(PvtolEnv, self).__init__()

        self.dynamics_mode = 'Pvtol'
        # Define action and observation space
        # They must be gym.spaces objects
        low_safe_bound = np.array([-3.5, -15.0])
        high_safe_bound = np.array([3.5, 15.0])
        self.action_space = spaces.Box(low=low_safe_bound, high=high_safe_bound)
        self.safe_action_space = spaces.Box(low=low_safe_bound, high=high_safe_bound)
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(11,))
        self.bds = np.array([[-7., -6.], [7., 6.]])
        np.random.seed(seed)

        # barrier signal to train the neural barrier certificate, treated as hyperparameters and therefore can be tuned for the learning process
        self.little_b = 0.1
        self.capital_b = -0.1

        self.get_f, self.get_g = self._get_dynamics()

        self.dt = 0.02
        self.max_episode_steps = 2000
        self.reward_goal = 1500.0
        self.goal_size = 3.5
        # Initialize Env
        self.state = None
        self.episode_step = 0
        self.initial_state = np.array([-4.5, -4.5, 0.0, 0.0, 0.0, 1.0, -4.5])  # x y theta v1 v2 safety-operator
        self.goal_pos = np.array([4.5, 4.5])
        self.safety_cost_coef = 1.0
        self.safety_operator_follow = 0.7
        self.y_min = -100.0
        self.y_max = 100.0



        # Build Hazards
        self.hazards = [{'radius': 0.25, 'location': np.array([-2.5, -2.5])},
                        {'radius': 0.25, 'location': np.array([-2.5, 2.5])},
                        {'radius': 0.25, 'location': np.array([0.0, -3.5])},
                        {'radius': 0.25, 'location': np.array([0.0, 3.5])},
                        {'radius': 0.25, 'location': np.array([-4.5, 0.0])}]
        self.is_safety_operator = True
        self.safety_operator = np.array([])  # x-position

        self.operator_dist = 1.0  # max distance PVTOL can be from the safety operator (if present)

        self.reset()

        # Process Hazards for cost checking
        self.hazard_locations = np.array(
            [[-2.5, -2.5], [-2.5, 2.5], [0.0, -3.5], [0.0, 3.5], [-4.5, 0.0]])
        self.hazards_radius = 0.25
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
        new_obs : ndarray
          The new observation with the following structure:
          [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        state, reward,constraint,barrier_signal, lya_pre_term, done, info = self._step(action)
        return self.get_obs(), reward,constraint,barrier_signal, lya_pre_term,self.get_obs(), done, info

    def _step(self, action):
        """

        Parameters
        ----------
        action

        Returns
        -------
        state : ndarray
            New internal state of the agent [x y theta x_d y_d thrust].
        reward : float
            Reward collected during this transition.
        done : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """

        lya_pre_term = self.get_obs()
        # Start with our prior for continuous time system x' = f(x) + g(x)u
        self.state_dynamics = self.state[:6]
        # lya_pre_term = self.state[:5]

        self.state_dynamics += self.dt * (self.get_f(self.state_dynamics) + self.get_g(self.state_dynamics) @ action)
        self.state[0] = self.state_dynamics[0]
        self.state[1] = self.state_dynamics[1]
        self.state[2] = self.state_dynamics[2]
        self.state[3] = self.state_dynamics[3]
        self.state[4] = self.state_dynamics[4]
        self.state[5] = self.state_dynamics[5]

        info = dict()

        # Update Position of Operator
        if self.is_safety_operator:
            self.safety_operator[0] = self.safety_operator[0] + self.safety_operator_follow * (self.state_dynamics[0] - self.safety_operator[0])
            self.state[6] = self.safety_operator[0]
        self.episode_step += 1

        # lya_next_term = self.state[:5]

        dist_goal = self._goal_dist()
        constraint = dist_goal
        # reward = (self.last_goal_dist - dist_goal)  # -1e-3 * dist_goal
        reward = -1e-3 * dist_goal
        self.last_goal_dist = dist_goal
        # Check if goal is met
        if self.goal_met():
            info['goal_met'] = True
            reward += self.reward_goal
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps

        barrier_signal = self.little_b

        # Constraint cost
        if self.hazards and np.any(np.sum((self.state[:2] - self.hazard_locations)**2, axis=1) < self.hazards_radius**2):
            num_obstacles = self.hazard_locations.shape[0]
            for i in range(num_obstacles):
                if np.sum((self.state[:2] - self.hazard_locations[i]) ** 2, axis=0) < self.hazards_radius ** 2:
                    barrier_signal = self.capital_b
                    # prYellow('Collision with an obstacle occured.')
                    if 'num_safety_violation_obstacles' in info:
                        info['num_safety_violation_obstacles'] += 1
                    else:
                        info['num_safety_violation_obstacles'] = 1

                    if 'safety_cost_obstacles' in info:
                        dist_to_center = np.sqrt(np.sum((self.state[:2] - self.hazard_locations[i])**2, axis=0))
                        safety_cost_val = ((self.hazards_radius - dist_to_center) / self.hazards_radius) * self.safety_cost_coef
                        info['safety_cost_obstacles'] += safety_cost_val
                    else:
                        dist_to_center = np.sqrt(np.sum((self.state[:2] - self.hazard_locations[i]) ** 2, axis=0))
                        safety_cost_val = ((self.hazards_radius - dist_to_center) / self.hazards_radius) * self.safety_cost_coef
                        info['safety_cost_obstacles'] = safety_cost_val


        if np.linalg.norm(self.state[0] - self.safety_operator[0]) >= self.operator_dist:
            if barrier_signal == self.little_b:
                barrier_signal = self.capital_b
            else:
                barrier_signal = barrier_signal + self.capital_b
            if self.state[0] - self.safety_operator[0] <= -self.operator_dist:
                # prRed('The agent is too left away from the operator.')
                if 'num_safety_violation_safety_operator' in info:
                    info['num_safety_violation_safety_operator'] += 1
                else:
                    info['num_safety_violation_safety_operator'] = 1

                safety_cost_operator_val = (-(self.state[0] - self.safety_operator[0]) - self.operator_dist) * self.safety_cost_coef
                if 'safety_cost_operator_val' in info:
                    info['safety_cost_operator_val'] += safety_cost_operator_val
                else:
                    info['safety_cost_operator_val'] = safety_cost_operator_val


            else:
                # prRed('The agent is too right away from the operator.')
                if 'num_safety_violation_safety_operator' in info:
                    info['num_safety_violation_safety_operator'] += 1
                else:
                    info['num_safety_violation_safety_operator'] = 1

                safety_cost_operator_val = ((
                            self.state[0] - self.safety_operator[0]) - self.operator_dist) * self.safety_cost_coef
                if 'safety_cost_operator_val' in info:
                    info['safety_cost_operator_val'] += safety_cost_operator_val
                else:
                    info['safety_cost_operator_val'] = safety_cost_operator_val

        if self.state[1] > self.y_max:
            if barrier_signal == self.little_b:
                barrier_signal = self.capital_b
            else:
                barrier_signal = barrier_signal + self.capital_b
            if 'num_safety_violation_y_max' in info:
                info['num_safety_violation_y_max'] += 1
            else:
                info['num_safety_violation_y_max'] = 1
            safety_cost_y_max_val = (self.state[1] - self.y_max) * self.safety_cost_coef
            if 'safety_cost_y_max_val' in info:
                info['safety_cost_y_max_val'] += safety_cost_y_max_val
            else:
                info['safety_cost_y_max_val'] = safety_cost_y_max_val

        if self.state[1] < self.y_min:
            if barrier_signal == self.little_b:
                barrier_signal = self.capital_b
            else:
                barrier_signal = barrier_signal + self.capital_b
            if 'num_safety_violation_y_min' in info:
                info['num_safety_violation_y_min'] += 1
            else:
                info['num_safety_violation_y_min'] = 1
            safety_cost_y_min_val = (self.y_min - self.state[1]) * self.safety_cost_coef
            if 'safety_cost_y_min_val' in info:
                info['safety_cost_y_min_val'] += safety_cost_y_min_val
            else:
                info['safety_cost_y_min_val'] = safety_cost_y_min_val



        return self.state, reward,constraint,barrier_signal,lya_pre_term, done, info

    def goal_met(self):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """

        return np.linalg.norm(self.state[:2] - self.goal_pos) <= self.goal_size

    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.episode_step = 0

        # Re-initialize state

        self.state = np.array([-4.5, -4.5, 0.0, 0.0, 0.0, 1.0, -4.5])

        if self.is_safety_operator:
            self.safety_operator = np.array([self.state[0]])  # x-position

        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist()

        return self.get_obs()

    def _get_dynamics(self):


        def get_f(state_dynamics):
            theta = state_dynamics[2]
            f_x = np.zeros(state_dynamics.shape)
            f_x[0] = state_dynamics[3]
            f_x[1] = state_dynamics[4]
            f_x[3] = -np.sin(theta) * state_dynamics[5]
            f_x[4] = np.cos(theta) * state_dynamics[5] - 1.0
            # f_x[4] = -1.0
            # f_x[4] = 0.0
            return f_x

        def get_g(state_dynamics):
            # theta = state_dynamics[2]
            g_x = np.array([[0, 0],
                            [0, 0],
                            [0, 1],
                            [0, 0],
                            [0, 0],
                            [1, 0]])
            return g_x

        return get_f, get_g



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
            print('Ep_step = {}, \tState = {}, \tDist2Goal = {}, alignment_error = {}'.format(self.episode_step, self.state, self._goal_dist(), theta_error))

        screen_width = 1300
        screen_height = 1000

        if self.viewer is None:
            from Pvtol_RL_training.envs import pyglet_rendering

            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            # Draw obstacles
            obstacles = []
            for i in range(len(self.hazards)):
                hazard_loc = self.hazards[i]['location']
                obstacles.append(pyglet_rendering.make_circle(radius=to_pixel(self.hazards[i]['radius'], shift=0), filled=True))
                obs_trans = pyglet_rendering.Transform(translation=(to_pixel(hazard_loc[0], shift=screen_width/2), to_pixel(hazard_loc[1], shift=screen_height/2)))
                obstacles[i].set_color(1.0, 0.0, 0.0)
                obstacles[i].add_attr(obs_trans)
                self.viewer.add_geom(obstacles[i])

            # Make Goal
            goal = pyglet_rendering.make_circle(radius=to_pixel(self.goal_size, shift=0), filled=True)
            goal_trans = pyglet_rendering.Transform(translation=(to_pixel(self.goal_pos[0], shift=screen_width/2), to_pixel(self.goal_pos[1], shift=screen_height/2)))
            goal.add_attr(goal_trans)
            goal.set_color(0.0, 0.5, 0.0)
            self.viewer.add_geom(goal)

            # Make Robot
            self.robot = pyglet_rendering.make_polygon(to_pixel(np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])), filled=True)
            self.robot_trans = pyglet_rendering.Transform(translation=(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2)))
            self.robot_trans.set_rotation(self.state[2])
            self.robot.add_attr(self.robot_trans)
            self.robot.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.robot)
            self.robot_orientation = pyglet_rendering.Line(start=(0.0, 0.0), end=(0.0, 15.0))
            self.robot_orientation.linewidth.stroke = 2
            self.robot_orientation.add_attr(self.robot_trans)
            self.robot_orientation.set_color(0, 0, 0)
            self.viewer.add_geom(self.robot_orientation)

            # Make Safety Operator
            if self.is_safety_operator:
                self.safety_operator_fh = pyglet_rendering.make_polygon(to_pixel(np.array([[-0.1, -0.2], [-0.1, 0.2], [0.1, 0.2], [0.1, -0.2]])), filled=True)
                self.safety_operator_fh.set_color(0.7, 0.7, 0.7)
                self.safety_operator_trans = pyglet_rendering.Transform(translation=(to_pixel(self.safety_operator[0], shift=screen_width/2), to_pixel(-5.8, shift=screen_height/2)))
                self.safety_operator_fh.add_attr(self.safety_operator_trans)
                self.viewer.add_geom(self.safety_operator_fh)
                self.safety_operator_right = pyglet_rendering.Line(start=(to_pixel(-self.operator_dist), to_pixel(self.bds[0][1]-5.7, shift=screen_height/2)), end=(to_pixel(-self.operator_dist), to_pixel(self.bds[1][1]-5.7, shift=screen_height/2)))
                self.safety_operator_left = pyglet_rendering.Line(start=(to_pixel(self.operator_dist), to_pixel(self.bds[0][1]-5.7, shift=screen_height/2)), end=(to_pixel(self.operator_dist), to_pixel(self.bds[1][1]-5.7, shift=screen_height/2)))
                self.safety_operator_right.add_attr(self.safety_operator_trans)
                self.safety_operator_left.add_attr(self.safety_operator_trans)
                self.viewer.add_geom(self.safety_operator_right)
                self.viewer.add_geom(self.safety_operator_left)


        if self.state is None:
            return None

        self.robot_trans.set_translation(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2))
        self.robot_trans.set_rotation(self.state[2])
        if self.is_safety_operator:
            self.safety_operator_trans.set_translation(to_pixel(self.safety_operator[0], shift=screen_width/2), to_pixel(-5.8, shift=screen_height/2))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [p_x, p_y, cos(theta), sin(theta), v_x, v_y, thrust, xdir2goal, ydir2goal, exp(-dist2goal)]
        """

        rel_loc = self.goal_pos - self.state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()  # compass to the goal

        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), self.state[3], self.state[4], self.state[5], self.state[6], goal_compass[0], goal_compass[1], np.exp(-goal_dist)])

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

    def _goal_dist(self):
        return np.linalg.norm(self.goal_pos - self.state[:2])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



