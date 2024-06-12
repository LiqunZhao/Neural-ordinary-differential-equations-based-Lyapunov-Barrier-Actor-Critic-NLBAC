""" This part is adapted from:
https://github.com/yemam3/Mod-RL-RCBF

"""
import numpy as np
import torch
from sac_cbf_clf.utils import to_tensor, to_numpy


class DynamicsModel:

    def __init__(self, env, args):
        """Constructor of DynamicsModel.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        """

        self.env = env

        self.device = torch.device("cuda" if args.cuda else "cpu")

    def get_state(self, obs):
        """Given the observation, this function does the pre-processing necessary and returns the state.

        Parameters
        ----------
        obs_batch : ndarray or torch.tensor
            Environment observation.

        Returns
        -------
        state_batch : ndarray or torch.tensor
            State of the system.

        """

        expand_dims = len(obs.shape) == 1
        is_tensor = torch.is_tensor(obs)

        if is_tensor:
            dtype = obs.dtype
            device = obs.device
            obs = to_numpy(obs)

        if expand_dims:
            obs = np.expand_dims(obs, 0)

        if self.env.dynamics_mode == 'Unicycle':
            theta = np.arctan2(obs[:, 3], obs[:, 2])
            state_batch = np.zeros((obs.shape[0], 3))
            state_batch[:, 0] = obs[:, 0]
            state_batch[:, 1] = obs[:, 1]
            state_batch[:, 2] = theta
        elif self.env.dynamics_mode == 'SimulatedCars':
            state_batch = np.copy(obs)
            state_batch[:, ::2] *= 100.0  # Scale Positions
            state_batch[:, 1::2] *= 30.0  # Scale Velocities
        else:
            raise Exception('Unknown dynamics')

        if expand_dims:
            state_batch = state_batch.squeeze(0)

        return to_tensor(state_batch, dtype, device) if is_tensor else state_batch

    def get_obs(self, state_batch):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------
        state : ndarray
            Environment state batch of shape (batch_size, n_s)

        Returns
        -------
        obs : ndarray
          Observation batch of shape (batch_size, n_o)

        """

        if self.env.dynamics_mode == 'Unicycle':
            obs = np.zeros((state_batch.shape[0], 4))
            obs[:, 0] = state_batch[:, 0]
            obs[:, 1] = state_batch[:, 1]
            obs[:, 2] = np.cos(state_batch[:, 2])
            obs[:, 3] = np.sin(state_batch[:, 2])
        elif self.env.dynamics_mode == 'SimulatedCars':
            obs = state_batch.clone()
            obs[:, ::2, 0] /= 100.0  # Scale Positions
            obs[:, 1::2, 0] /= 30.0  # Scale Velocities
        else:
            raise Exception('Unknown dynamics')
        return obs


    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

