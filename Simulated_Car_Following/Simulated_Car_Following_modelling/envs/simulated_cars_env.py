""" This environment is adapted from:
https://github.com/yemam3/Mod-RL-RCBF

"""
import numpy as np
import gym
from gym import spaces

class SimulatedCarsEnv(gym.Env):
    """Simulated Car Following Env,
    Front <- Car 1 <- Car 2 <- Car 3 <- Car 4 (controlled) <- Car 5
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(SimulatedCarsEnv, self).__init__()

        self.dynamics_mode = 'SimulatedCars'
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,))
        self.safe_action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,))  # be the same with the action space.
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(10,))
        self.max_episode_steps = 300
        self.dt = 0.02

        # Gains
        self.kp = 4.0
        self.k_brake = 20.0

        self.state = None  # State [x_1 v_1 ... x_5 v_5]
        self.t = 0  # Time
        self.episode_step = 0   # Episode Step
        self.should_keep = 9.5  # The desired distance between the 3rd and 4th cars
        self.should_keep_thre = 0.5  # If the distance between the 3rd and 4th cars ranges from 9 to 10, then grant it an additional reward
        self.reward_goal = 2.0       # The additional reward

        self.reset()  # initialize the env
        self.safety_cost_coef = 1.0

    def step(self, action):
        """

        Parameters
        ----------
        action

        Returns
        -------
        obs : ndarray
            New observation of the agent.
        reward : float
            Reward collected during this transition.
        constraint: float
            Constraint collected as cost for approximating the Lyapunov function
        previous_positions: ndarray
            Position and velocities of the 3rd and 4th cars before the action is taken. This will be used as the input of the Lyapunov network.
        next_positions: ndarray
            Position and velocities of the 3rd and 4th cars after the action is taken. This will be used as the input of the Lyapunov network.
        done : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """

        # Current State
        pos = self.state[::2]
        vels = self.state[1::2]

        self.record_action = action

        # Accelerations
        vels_des = 3.0 * np.ones(5)  # a pre-defined velocities
        vels_des[0] -= 4 * np.sin(self.t)  # modify the desired velocity of the first car
        # calculate the accelerations of 5 cars
        accels = self.kp * (
                    vels_des - vels)
        accels[1] += -self.k_brake * (pos[0] - pos[1]) * (
                    (pos[0] - pos[1]) < 6.5)
        accels[2] += -self.k_brake * (pos[1] - pos[2]) * ((pos[1] - pos[2]) < 6.5)
        accels[3] = 0.0
        accels[4] += -self.k_brake * (pos[2] - pos[4]) * (
                    (pos[2] - pos[4]) < 13.0)

        # Unknown part to the dynamics
        accels *= 1.1

        self.previous_positions = np.array([self.state[4], self.state[5],self.state[6],self.state[7]])


        f_x = np.zeros(10)
        g_x = np.zeros(10)

        f_x[::2] = vels  # Derivatives of positions are velocities
        f_x[1::2] = accels  # Derivatives of velocities are acceleration
        f_x[7] = 0.0
        g_x[7] = 1.0  # Car 4's acceleration is the control input

        self.state += self.dt * (f_x + g_x * action)
        self.t = self.t + self.dt  # time
        self.episode_step += 1  # steps in episode

        info = dict()

        self.third_pos = self.state[4]
        self.fourth_pos = self.state[6]
        self.next_positions = np.array([self.state[4], self.state[5], self.state[6], self.state[7]])
        self.fifth_pos = self.state[8]

        self.distance_third_fourth = self.third_pos - self.fourth_pos

        reward = self._get_reward(action[0])

        satisfied_num = 0
        if (abs(self.distance_third_fourth - self.should_keep) < self.should_keep_thre):
            satisfied_num = 1                                     # The desired region is reached
            reward = reward + self.reward_goal                    # Additional reward for reaching the region [9.0,10.0]

        info['reached'] = satisfied_num


        done = self.episode_step >= self.max_episode_steps  # done?

        info['goal_met'] = False

        # Include the cost in info
        num_safety_violation = 0
        safety_cost_val = 0
        if ((self.third_pos - self.fourth_pos) < 2.5):
            num_safety_violation = num_safety_violation + 1
            dist_third_fourth = (self.third_pos - self.fourth_pos)
            safety_cost_val = safety_cost_val + np.abs(dist_third_fourth - 2.5) * self.safety_cost_coef

        if ((self.fourth_pos - self.fifth_pos) < 2.5):
            num_safety_violation = num_safety_violation + 1
            dist_fourth_fourth = (self.fourth_pos - self.fifth_pos)
            safety_cost_val = safety_cost_val + np.abs(dist_fourth_fourth - 2.5) * self.safety_cost_coef

        info['num_safety_violation'] = num_safety_violation
        info['safety_cost'] = safety_cost_val                            # Used by other algorithms like CPO, PPO-Lag and TRPO-Lag


        constraint = abs(self.distance_third_fourth - self.should_keep)

        return self._get_obs(), reward, constraint, self.previous_positions, self.next_positions, done, info

    def _get_reward(self, action):
        """
        Reward function
        """

        self.reward_action = -0.5 * np.abs(action**2) / self.max_episode_steps

        return self.reward_action


    def reset(self):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """

        self.t = 0
        self.state = np.zeros(10)  # first col is pos, 2nd is vel
        self.state[::2] = [42.0, 34.0, 26.0, 18.0, 10.0]  # initial positions
        self.state[1::2] = 3.0 + np.random.normal(0, 0.5)  # initial velocities
        self.state[7] = 3.0  # initial velocity of car 4


        self.episode_step = 0
        self.num_conti_follow = 0

        return self._get_obs()

    def render(self, mode='human', close=False):
        """Render the environment to the screen

        Parameters
        ----------
        mode : str
        close : bool

        Returns
        -------

        """

        print('Ep_step = {}, action={}, first = {:.4f},{:.4f}, third = {:.4f},{:.4f}, fourth = {:.4f},{:.4f}, fifth = {:.4f},{:.4f}, diff1 = {:.4f}, diff2 = {:.4f}, reward_act = {:.4f}'.format(self.episode_step, self.record_action,self.state[0],self.state[1], self.state[4],self.state[5],self.state[6],self.state[7],self.state[8],self.state[9],
                                                                                                       self.state[4]-self.state[6],self.state[6]-self.state[8],self.reward_action))


    def _get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [car_1_x, car_1_v, car_1_a, ...]
        """

        obs = np.copy(np.ravel(self.state))
        obs[::2] /= 100.0  # scale positions
        obs[1::2] /= 30.0  # scale velocities
        return obs
