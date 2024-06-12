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

        if self.env.dynamics_mode == 'Pvtol':
            state_batch = np.zeros((obs.shape[0], 7))
            theta = np.arctan2(obs[:, 3], obs[:, 2])
            state_batch[:, 0] = obs[:, 0]
            state_batch[:, 1] = obs[:, 1]
            state_batch[:, 2] = theta
            state_batch[:, 3] = obs[:, 4]
            state_batch[:, 4] = obs[:, 5]
            state_batch[:, 5] = obs[:, 6]
            state_batch[:, 6] = obs[:, 7]
        else:
            raise Exception('Unknown dynamics')

        state_dynamics_batch = state_batch[:, :6]

        if expand_dims:
            state_batch = state_batch.squeeze(0)
            state_dynamics_batch = state_dynamics_batch.squeeze(0)

        if is_tensor:
            return to_tensor(state_batch, dtype, device), to_tensor(state_dynamics_batch, dtype, device)
        else:
            return state_batch, state_dynamics_batch

    def get_obs(self, state_batch,device):
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

        # if self.env.dynamics_mode == 'Unicycle':
        #     obs = np.zeros((state_batch.shape[0], 4))
        #     obs[:, 0] = state_batch[:, 0]
        #     obs[:, 1] = state_batch[:, 1]
        #     obs[:, 2] = np.cos(state_batch[:, 2])
        #     obs[:, 3] = np.sin(state_batch[:, 2])
        # elif self.env.dynamics_mode == 'SimulatedCars':
        #     obs = np.copy(state_batch)
        #     obs[:, ::2] /= 100.0  # Scale Positions
        #     obs[:, 1::2] /= 30.0  # Scale Velocities
        if self.env.dynamics_mode == 'Pvtol':
            obs = torch.zeros((state_batch.shape[0], 11, 1)).to(device)
            thetas = state_batch[:, 2, :].squeeze(-1)
            c_thetas = torch.cos(thetas)
            s_thetas = torch.sin(thetas)
            obs[:, 0, 0] = state_batch[:, 0, :].squeeze(-1)
            obs[:, 1, 0] = state_batch[:, 1, :].squeeze(-1)
            obs[:, 2, 0] = c_thetas
            obs[:, 3, 0] = s_thetas
            obs[:, 4, 0] = state_batch[:, 3, :].squeeze(-1)
            obs[:, 5, 0] = state_batch[:, 4, :].squeeze(-1)
            obs[:, 6, 0] = state_batch[:, 5, :].squeeze(-1)
            obs[:, 7, 0] = state_batch[:, 6, :].squeeze(-1)

            goal_pos = torch.zeros((state_batch.shape[0], 2, 1)).to(device)
            goal_pos[:, 0, 0] = 4.5
            goal_pos[:, 1, 0] = 4.5
            rel_loc = goal_pos - state_batch[:, :2, :]
            rel_loc_squeeze = rel_loc.squeeze(-1)
            goal_dist = torch.norm(rel_loc_squeeze,dim=1)

            Rs = torch.zeros((state_batch.shape[0],2,2)).to(device)
            Rs[:, 0, 0] = c_thetas
            Rs[:, 0, 1] = s_thetas
            Rs[:, 1, 0] = -s_thetas
            Rs[:, 1, 1] = c_thetas

            # rel_loc_reshape = rel_loc.reshape((state_batch.shape[0], 1, 2))

            # vecs = torch.bmm(rel_loc_reshape,Rs)
            vecs = torch.bmm(Rs, rel_loc)

            vecs_squeeze = vecs.squeeze()

            # print(vecs.shape)
            div = torch.norm(vecs_squeeze,dim=1)
            # print(div.shape)

            # vecs /= torch.norm(vecs,dim=2).squeeze(-1) + 0.001

            vecs_res = torch.zeros((state_batch.shape[0], 2, 1)).to(device)
            for i in range(state_batch.shape[0]):
                vecs_res[i, 0, 0] = vecs[i, 0, 0] / (div[i] + 0.001)
                vecs_res[i, 1, 0] = vecs[i, 1, 0] / (div[i] + 0.001)

            obs[:, 8, 0] = vecs_res[:, 0, :].squeeze(-1)
            obs[:, 9, 0] = vecs_res[:, 1, :].squeeze(-1)
            obs[:, 10, 0] = torch.exp(-goal_dist)

            # print('The example observation from the predicted state is')
            # print(obs[0, :, :])
        else:
            raise Exception('Unknown dynamics')
        return obs


    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

