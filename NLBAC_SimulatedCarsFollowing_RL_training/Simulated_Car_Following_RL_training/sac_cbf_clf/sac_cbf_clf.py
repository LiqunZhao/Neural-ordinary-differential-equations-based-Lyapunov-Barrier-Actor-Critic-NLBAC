import random
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac_cbf_clf.utils import soft_update, hard_update
from sac_cbf_clf.model import GaussianPolicy, QNetwork, DeterministicPolicy, LyaNetwork, NeuralODEModel, train_step
import numpy as np
from sac_cbf_clf.utils import to_tensor
from torchdiffeq import odeint
import torch.nn as nn
DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2},
                 'SimulatedCars': {'n_s': 10, 'n_u': 1}}


class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, predicted_state, true_state):
        mse_loss = nn.MSELoss(reduction='mean')
        L = mse_loss(predicted_state, true_state)
        return L


class SAC_CBF_CLF(object):

    def __init__(self, num_inputs, action_space, env, args):
        self.gamma = args.gamma
        self.gamma_b = args.gamma_b
        self.tau = args.tau
        self.alpha = args.alpha
        self.backup_alpha = args.alpha
        self.lyapunov_network_input_number = 4   # How many elements the input of the Lyapunov network used as CLF has

        self.policy_type = args.policy
        self.batch_size = args.batch_size
        self.target_update_interval = args.target_update_interval
        self.Lagrangian_multiplier_update_interval = args.Lagrangian_multiplier_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.action_space = action_space
        self.action_space.seed(args.seed)
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.critic_lyapunov_lr = 0.0004

        # Value and Lyapunov networks
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lyapunov_lr)

        self.lyapunovNet = LyaNetwork(self.lyapunov_network_input_number,args.hidden_size).to(device=self.device)
        self.lyaNet_optim = Adam(self.lyapunovNet.parameters(),lr = self.critic_lyapunov_lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.lyapunovNet_target = LyaNetwork(self.lyapunov_network_input_number,args.hidden_size).to(device=self.device)
        hard_update(self.lyapunovNet_target, self.lyapunovNet)

        self.cost_limit = 0.0
        self.augmented_term = 1.0    # initial value of the coefficient of the augmented squared terms
        self.augmented_ratio = 1.0005    # the ratio to increase the coefficient

        # Set the random seed
        if args.seed >= 0:
            env.seed(args.seed)
            random.seed(args.seed)
            env.action_space.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # The controllers
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

                self.backup_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.backup_alpha_optim = Adam([self.backup_log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

            self.backup_policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.backup_policy_optim = Adam(self.backup_policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        # CBF layer
        self.env = env

        if self.env.dynamics_mode not in DYNAMICS_MODE:
            raise Exception('Dynamics mode not supported.')

        elif self.env.dynamics_mode == 'SimulatedCars':
            self.num_cbfs = 2

        self.action_dim = env.action_space.shape[0]

        self.u_min, self.u_max = self.get_control_bounds()

        num_cbfs = self.num_cbfs
        num_clfs = 1

        self.num_constraints = num_cbfs + num_clfs

        # Lagrangian multipliers
        self.lambda_values = []
        for i in range(self.num_constraints):
            lambda_value = 0.0
            self.lambda_values.append(lambda_value)

        self.backup_lambda_values = []
        for i in range(self.num_cbfs):
            backup_lambda_value = 0.0
            self.backup_lambda_values.append(backup_lambda_value)

        # The NODE model. Remember to change the parameter when necessary
        self.neural_ode_model = NeuralODEModel(12, 10)
        self.neural_ode_model.to(self.device)
        self.solver = 'euler'
        self.neural_ode_model_optimizer = torch.optim.Adam(self.neural_ode_model.parameters(), lr=1e-3)
        self.model_loss_func = PoseLoss()

    def select_action(self, state, evaluate=False, warmup=False):

        state = to_tensor(state, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0)

        if warmup:             # Sample the actions from the action space randomly
            batch_size = state.shape[0]
            action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
            for i in range(batch_size):
                action[i] = torch.from_numpy(self.action_space.sample()).to(self.device)
        else:
            if evaluate is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)


        action = action.detach().cpu().numpy()[0] if expand_dim else action.detach().cpu().numpy()

        return action

    def select_action_backup(self, state, evaluate=False, warmup=False):
        state = to_tensor(state, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0)

        if warmup:
            batch_size = state.shape[0]
            action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
            for i in range(batch_size):
                action[i] = torch.from_numpy(self.action_space.sample()).to(self.device)
        else:
            if evaluate is False:
                action, _, _ = self.backup_policy.sample(state)
            else:
                _, _, action = self.backup_policy.sample(state)

        action = action.detach().cpu().numpy()[0] if expand_dim else action.detach().cpu().numpy()
        return action



    def update_parameters(self, memory, batch_size, updates, dynamics_model, NODE_memory,NODE_model_update_interval):
        """
        Update parameters of the RL-based controllers

        Parameters
        ----------
        memory : ReplayMemory
        batch_size : int
        updates : int
        dynamics_model : The dynamics model, which helps to convert obs to state
        NODE_memory: ReplayMemory
        NODE_model_update_interval: int

        Returns
        -------
        Some information about the losses

        """

        # Sample the data to train the RL-based controller
        state_batch, action_batch, reward_batch, constraint_batch, previous_positions_batch, next_positions_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(
            batch_size=batch_size)

        # Update the NODE modelling of the dynamics by using the data collected during the RL-based controller training
        if updates % NODE_model_update_interval == 0:
            NODE_batch_size = min(NODE_memory.position, 32768)     # Set an upper bound for the number of data used to train the NODE modelling
            node_obs_batch, node_action_batch, node_reward_batch, node_constraint_batch, node_previous_positions_batch, node_next_positions_batch, node_next_obs_batch, node_mask_batch, node_t_batch, node_next_t_batch = NODE_memory.sample(
                batch_size=NODE_batch_size)
            node_obs_batch = torch.FloatTensor(node_obs_batch).to(self.device)
            node_next_obs_batch = torch.FloatTensor(node_next_obs_batch).to(self.device)
            node_action_batch = torch.FloatTensor(node_action_batch).to(self.device)
            node_t_batch = torch.FloatTensor(node_t_batch).to(self.device).unsqueeze(1)
            node_state_batch = dynamics_model.get_state(node_obs_batch)
            node_next_state_batch = dynamics_model.get_state(node_next_obs_batch)

            average_train_loss = train_step(model=self.neural_ode_model, state=node_state_batch,
                                            action=node_action_batch, next_state=node_next_state_batch, time_batch = node_t_batch,
                                            optimizer=self.neural_ode_model_optimizer,
                                            loss_func=self.model_loss_func, horizon=NODE_batch_size,
                                            solver=self.solver, time_interval=self.env.dt)
        # The data to train the RL-based controller
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(self.device).unsqueeze(1)
        previous_positions_batch = torch.FloatTensor(previous_positions_batch).to(self.device)
        next_positions_batch = torch.FloatTensor(next_positions_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        time_batch = torch.FloatTensor(t_batch).to(self.device).unsqueeze(1)
        next_time_batch = torch.FloatTensor(next_t_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

            lf_next_target = self.lyapunovNet_target(next_positions_batch)
            next_l_value = constraint_batch + mask_batch * self.gamma * (lf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        lf = self.lyapunovNet(previous_positions_batch)
        lf_loss = F.mse_loss(lf, next_l_value)


        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        self.lyaNet_optim.zero_grad()
        lf_loss.backward()
        self.lyaNet_optim.step()

        # Compute Actions and log probabilities
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Do the same things for training the RL-based backup controller
        backup_pi, backup_log_pi, _ = self.backup_policy.sample(state_batch)

        backup_qf1_pi, backup_qf2_pi = self.critic(state_batch, backup_pi)
        backup_min_qf_pi = torch.min(backup_qf1_pi, backup_qf2_pi)

        # The part without CBF or CLF constraints for the primary controller
        policy_loss_1 = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # The part without CBF constraints for the backup controller
        backup_policy_loss_1 = ((self.backup_alpha * backup_log_pi) - backup_min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # The CBF and CLF constraints part for the primary controller
        policy_loss_2 = self.get_cbf_clf_part(state_batch, pi, dynamics_model, previous_positions_batch,time_batch,next_time_batch,updates)  # Get other terms in the augmented Lagrangian function

        # The CBF constraints part for the backup controller
        backup_policy_loss_2 = self.backup_get_cbf_clf_part(
            state_batch, backup_pi, dynamics_model, time_batch,next_time_batch,updates)

        policy_loss = (policy_loss_1 + policy_loss_2)
        backup_policy_loss = (backup_policy_loss_1 + backup_policy_loss_2)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.backup_policy_optim.zero_grad()
        backup_policy_loss.backward()
        self.backup_policy_optim.step()


        if self.automatic_entropy_tuning:     # Update the temperature parameter
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()

            backup_alpha_loss = -(self.backup_log_alpha * (backup_log_pi + self.target_entropy).detach()).mean()

            self.backup_alpha_optim.zero_grad()
            backup_alpha_loss.backward()
            self.backup_alpha_optim.step()

            self.backup_alpha = self.backup_log_alpha.exp()

        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        # Soft update
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.lyapunovNet_target, self.lyapunovNet, self.tau)

        return qf1_loss.item(), qf2_loss.item(),lf_loss.item(), policy_loss_1.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, output):
        print('Saving models in {}'.format(output))
        torch.save(
            self.policy.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
        torch.save(
            self.lyapunovNet.state_dict(),
            '{}/lyapunov.pkl'.format(output)
        )
        torch.save(
            self.neural_ode_model.state_dict(),
            '{}/node_model.pkl'.format(output)
        )

    # Load model parameters
    def load_weights(self, output):
        if output is None: return
        print('Loading models from {}'.format(output))

        self.policy.load_state_dict(
            torch.load('{}/actor.pkl'.format(output), map_location=torch.device(self.device))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output), map_location=torch.device(self.device))
        )
        self.lyapunovNet.load_state_dict(
            torch.load('{}/lyapunov.pkl'.format(output), map_location=torch.device(self.device))
        )

    def load_model(self, actor_path, critic_path,lyapunov_path):
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if lyapunov_path is not None:
            self.lyapunovNet.load_state_dict(torch.load(lyapunov_path))

    def get_cbf_clf_part(self, obs_batch, action_batch, dynamics_model,previous_positions_batch,time_batch,next_time_batch,updates):
        """Calculate the value of CBF and CLF constraints part for the primary controller.

        Parameters
        ----------
        obs_batch : torch.tensor
        action_batch : torch.tensor
        dynamics_model : DynamicsModel
        previous_positions_batch : torch.tensor
        time_batch : torch.tensor
        next_time_batch : torch.tensor
        updates : int

        Returns
        -------
        policy_loss_2 : The value of CBF and CLF constraints part for the primary controller
        """

        state_batch = dynamics_model.get_state(obs_batch)     #convert obs to state
        previous_positions_batch = previous_positions_batch.requires_grad_()
        lyapunov_value = self.lyapunovNet(previous_positions_batch)   # Here use previous_positions_batch instead of state_batch as the inputs of the Lyapunov network (since we know the constraint is used to try to achieve a desired distance between the 3rd and 4th cars)
        lyapunov_value_detach = lyapunov_value.detach()
        policy_loss_2 = self.get_policy_loss_2(dynamics_model,state_batch, action_batch,lyapunov_value_detach,time_batch,next_time_batch,updates)

        return policy_loss_2

    def backup_get_cbf_clf_part(self, obs_batch, action_batch, dynamics_model,time_batch,next_time_batch,updates):   #currently write like this, used when only use QP for finding the one-time action for one-time rollout
        """Calculate the value of CBF constraints part for the backup controller.

        Parameters
        ----------
        obs_batch : torch.tensor
        action_batch : torch.tensor
        dynamics_model : DynamicsModel
        time_batch : torch.tensor
        next_time_batch : torch.tensor
        updates : int

        Returns
        -------
        policy_loss_2 : The value of CBF constraints part for the backup controller
        """

        state_batch = dynamics_model.get_state(obs_batch)
        backup_policy_loss_2 = self.backup_get_policy_loss_2(dynamics_model,state_batch, action_batch,time_batch,next_time_batch,updates)

        return backup_policy_loss_2

    def get_policy_loss_2(self, dynamics_model,state_batch, action_batch,lyapunov_value,t_batch,next_time_batch,updates):
        assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(t_batch.shape) == 2, print(state_batch.shape,
                                                                                    action_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b
        gamma_l = 0.15  # define the coefficient for the CLF

        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)
        t_batch = torch.unsqueeze(t_batch, -1)
        next_time_batch = torch.unsqueeze(next_time_batch, -1)

        if self.env.dynamics_mode == 'SimulatedCars':     # Make sure the env is correct
            collision_radius = 4.5
            state_batch_squeeze = state_batch.squeeze(-1)
            action_batch_squeeze = action_batch.squeeze(-1)
            t_batch_squeeze = t_batch.squeeze(-1)
            next_time_batch_squeeze = next_time_batch.squeeze(-1)
            action_time_batch_squeeze = torch.cat((action_batch_squeeze, t_batch_squeeze), dim=-1)
            model_input_t = torch.cat((state_batch_squeeze, action_time_batch_squeeze), dim=-1).to(device=self.device)
            t_span = torch.tensor([0, self.env.dt]).to(device=self.device)

            # Predict the next state x_(t+1) based on the NODE model
            next_state_pred = odeint(self.neural_ode_model, model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:,:10]

            next_state_pred_unsqueeze = next_state_pred.unsqueeze(-1)
            # convert state to obs for sampling the next action u_(t+1)
            next_obs_pred = dynamics_model.get_obs(next_state_pred_unsqueeze)
            next_obs_pred_detach = next_obs_pred.detach()
            next_obs_pred_detach_squeeze = next_obs_pred_detach.squeeze(-1)
            # sample the next action u_(t+1)
            pi_next, log_pi_next, _ = self.policy.sample(next_obs_pred_detach_squeeze)

            if len(pi_next.shape) == 2:
                pi_next = torch.unsqueeze(pi_next, -1)

            pi_next = pi_next.detach()   # Do not need the gradient information of u_(t+1) since only u_t is useful in CBF and CLF constraints
            pi_next_squeeze = pi_next.squeeze(-1)
            next_action_time_batch_squeeze = torch.cat((pi_next_squeeze, next_time_batch_squeeze), dim=-1)
            next_model_input_t = torch.cat((next_state_pred, next_action_time_batch_squeeze), dim=-1).to(
                device=self.device)

            # Predict the next next state x_(t+2) based on the NODE model
            next_next_state_pred = odeint(self.neural_ode_model, next_model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:, :10]
            next_next_state_pred_unsqueeze = next_next_state_pred.unsqueeze(-1)

            matrix_ten_to_four = torch.zeros((batch_size, 4, 10)).to(self.device)
            matrix_ten_to_four[:, 0, 4] = 1.0
            matrix_ten_to_four[:, 1, 5] = 1.0
            matrix_ten_to_four[:, 2, 6] = 1.0
            matrix_ten_to_four[:, 3, 7] = 1.0

            # Obtain the positions and velocities of the 3rd and 4th cars as the inputs of Lyapunov network
            lyapunov_next_state = torch.bmm(matrix_ten_to_four,next_state_pred_unsqueeze).squeeze(-1)

            lyapunov_value_next = self.lyapunovNet(lyapunov_next_state)

            # CLF constraint
            lya_term = (lyapunov_value_next - lyapunov_value) + gamma_l * lyapunov_value

            # To calculate the distance between the 3rd and 4th cars
            P23_star = torch.zeros(batch_size, 1, state_batch.shape[1]).to(self.device)
            P23_star[:, 0, 4] = 1.0
            P23_star[:, 0, 6] = -1.0

            # To calculate the distance between the 4th and 5th cars
            P34_star = torch.zeros(batch_size, 1, state_batch.shape[1]).to(self.device)
            P34_star[:, 0, 6] = 1.0
            P34_star[:, 0, 8] = -1.0

            q = torch.zeros(batch_size, 1, 1).to(self.device)
            q[:, 0, 0] = -collision_radius

            h23_next = (torch.bmm(P23_star, next_state_pred_unsqueeze) + q).squeeze(-1)
            h34_next = (torch.bmm(P34_star, next_state_pred_unsqueeze) + q).squeeze(-1)

            h23_now = (torch.bmm(P23_star, state_batch) + q).squeeze(-1)
            h34_now = (torch.bmm(P34_star, state_batch) + q).squeeze(-1)

            h23_next_next = (torch.bmm(P23_star, next_next_state_pred_unsqueeze) + q).squeeze(-1)
            h34_next_next = (torch.bmm(P34_star, next_next_state_pred_unsqueeze) + q).squeeze(-1)

            l1_cbf23_term = h23_next - h23_now + gamma_b * h23_now

            l1_cbf34_term = h34_next - h34_now + gamma_b * h34_now

            l2_cbf23_term = h23_next_next - h23_next + gamma_b * h23_next

            l2_cbf34_term = h34_next_next - h34_next + gamma_b * h34_next

            # High-order CBF constraints (relative-degree is 2)
            cbf23_term = -(l2_cbf23_term - l1_cbf23_term) - gamma_b * l1_cbf23_term

            cbf34_term = -(l2_cbf34_term - l1_cbf34_term) - gamma_b * l1_cbf34_term

        else:
            raise Exception('Dynamics mode unknown!')

        # Combine the CBF and CLF constraints within one matrix
        matr = torch.cat((cbf23_term,cbf34_term,lya_term),1)  # Obtain the matrix containing all the CBF and CLF information
        matr = matr.unsqueeze(-1)

        # Filter the matrix and only leave the ones whose values are larger than 0 (all the elements whose values are smaller than 0, which means CBF and CLF constraints have been satisfied at those states, are abandoned and replaced by 0 without gradient information). Similar to Relu function.
        filter = torch.zeros_like(matr)
        filtered_matr = torch.where(matr > 0, matr, filter)

        # Calculate the average values as the expected values
        required_matrix = torch.sum(filtered_matr, dim=0, keepdim=False)
        for i in range(required_matrix.shape[0]):
            required_matrix[i] = required_matrix[i] / self.batch_size

        # Here calculate a ratio that is used to try to keep a balance between the CBF and CLF constraint losses
        other_compoenent = required_matrix[:-1, :] - self.cost_limit
        other_compoenent_mean = torch.abs(torch.mean(other_compoenent))
        lya_component = torch.abs(required_matrix[-1, :] - self.cost_limit)
        ratio = float(other_compoenent_mean / lya_component)

        if ratio < 0.002:        # Even want to keep the balance, the ratio cannot be too small.
            ratio = 0.002

        required_matrix_copy = required_matrix.detach()

        # Update Lagrangian multipliers
        if updates % self.Lagrangian_multiplier_update_interval == 0:
            for i in range(self.num_constraints):
                previous_lambda = self.lambda_values[i]
                new_lambda = previous_lambda + self.augmented_term * required_matrix_copy[i]
                real_new_lambda = torch.clamp(new_lambda, 0.01, 300.0)        # Here set an upper bound for numerical stability
                self.lambda_values[i] = real_new_lambda

        # Update the coefficient of the augmented squared terms
        self.augmented_term = self.augmented_term * self.augmented_ratio
        self.augmented_term = min(self.augmented_term,200)                    # Here set an upper bound for numerical stability

        policy_loss_2 = float(self.lambda_values[0]) * (required_matrix[0] - self.cost_limit) + self.augmented_term / 2.0 * (required_matrix[0] - self.cost_limit) * (required_matrix[0] - self.cost_limit)
        for i in range(required_matrix.shape[0] - 2):
            policy_loss_2 += float(self.lambda_values[i + 1]) * (required_matrix[i + 1] - self.cost_limit) + self.augmented_term / 2.0 * (required_matrix[i + 1] - self.cost_limit) * (required_matrix[i + 1] - self.cost_limit)
        policy_loss_2 += float(self.lambda_values[-1]) * ratio * (required_matrix[-1] - self.cost_limit) + ratio * ratio * self.augmented_term / 2.0 * (required_matrix[-1] - self.cost_limit) * (required_matrix[-1] - self.cost_limit)

        return policy_loss_2

    def backup_get_policy_loss_2(self, dynamics_model, state_batch, action_batch, t_batch, next_time_batch, updates):
        assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(t_batch.shape) == 2, print(state_batch.shape, action_batch.shape)
        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b

        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)
        t_batch = torch.unsqueeze(t_batch, -1)
        next_time_batch = torch.unsqueeze(next_time_batch, -1)

        if self.env.dynamics_mode == 'SimulatedCars':     # Make sure the env is correct
            collision_radius = 4.5
            state_batch_squeeze = state_batch.squeeze(-1)
            action_batch_squeeze = action_batch.squeeze(-1)
            t_batch_squeeze = t_batch.squeeze(-1)
            next_time_batch_squeeze = next_time_batch.squeeze(-1)
            action_time_batch_squeeze = torch.cat((action_batch_squeeze, t_batch_squeeze), dim=-1)
            model_input_t = torch.cat((state_batch_squeeze, action_time_batch_squeeze), dim=-1).to(
                device=self.device)
            t_span = torch.tensor([0, self.env.dt]).to(device=self.device)

            # Predict the next state x_(t+1) based on the NODE model
            next_state_pred = \
            odeint(self.neural_ode_model, model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:, :10]

            next_state_pred_unsqueeze = next_state_pred.unsqueeze(-1)
            # convert state to obs for sampling the next action u_(t+1)
            next_obs_pred = dynamics_model.get_obs(next_state_pred_unsqueeze)
            next_obs_pred_detach = next_obs_pred.detach()
            next_obs_pred_detach_squeeze = next_obs_pred_detach.squeeze(-1)
            # sample the next action u_(t+1)
            pi_next, log_pi_next, _ = self.backup_policy.sample(next_obs_pred_detach_squeeze)

            if len(pi_next.shape) == 2:
                pi_next = torch.unsqueeze(pi_next, -1)

            pi_next = pi_next.detach()   # Do not need the gradient information of u_(t+1) since only u_t is useful in CBF and CLF constraints
            pi_next_squeeze = pi_next.squeeze(-1)
            next_action_time_batch_squeeze = torch.cat((pi_next_squeeze, next_time_batch_squeeze), dim=-1)
            next_model_input_t = torch.cat((next_state_pred, next_action_time_batch_squeeze), dim=-1).to(
                device=self.device)

            # Predict the next next state x_(t+2) based on the NODE model
            next_next_state_pred = \
            odeint(self.neural_ode_model, next_model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:, :10]
            next_next_state_pred_unsqueeze = next_next_state_pred.unsqueeze(-1)

            # To calculate the distance between the 3rd and 4th cars
            P23_star = torch.zeros(batch_size, 1, state_batch.shape[1]).to(self.device)
            P23_star[:, 0, 4] = 1.0
            P23_star[:, 0, 6] = -1.0

            # To calculate the distance between the 4th and 5th cars
            P34_star = torch.zeros(batch_size, 1, state_batch.shape[1]).to(self.device)
            P34_star[:, 0, 6] = 1.0
            P34_star[:, 0, 8] = -1.0

            q = torch.zeros(batch_size, 1, 1).to(self.device)
            q[:, 0, 0] = -collision_radius

            h23_next = (torch.bmm(P23_star, next_state_pred_unsqueeze) + q).squeeze(-1)
            h34_next = (torch.bmm(P34_star, next_state_pred_unsqueeze) + q).squeeze(-1)

            h23_now = (torch.bmm(P23_star, state_batch) + q).squeeze(-1)
            h34_now = (torch.bmm(P34_star, state_batch) + q).squeeze(-1)

            h23_next_next = (torch.bmm(P23_star, next_next_state_pred_unsqueeze) + q).squeeze(-1)
            h34_next_next = (torch.bmm(P34_star, next_next_state_pred_unsqueeze) + q).squeeze(-1)

            l1_cbf23_term = h23_next - h23_now + gamma_b * h23_now

            l1_cbf34_term = h34_next - h34_now + gamma_b * h34_now

            l2_cbf23_term = h23_next_next - h23_next + gamma_b * h23_next

            l2_cbf34_term = h34_next_next - h34_next + gamma_b * h34_next

            # High-order CBF constraints (relative-degree is 2)
            cbf23_term = -(l2_cbf23_term - l1_cbf23_term) - gamma_b * l1_cbf23_term

            cbf34_term = -(l2_cbf34_term - l1_cbf34_term) - gamma_b * l1_cbf34_term

        else:
            raise Exception('Dynamics mode unknown!')

        # Combine the CBF and CLF constraints within one matrix
        matr = torch.cat((cbf23_term, cbf34_term),1)  # Obtain the matrix containing all the CBF information
        matr = matr.unsqueeze(-1)

        # Filter the matrix and only leave the ones whose values are larger than 0 (all the elements whose values are smaller than 0, which means CBF constraints have been satisfied at those states, are abandoned and replaced by 0 without gradient information). Similar to Relu function.
        filter = torch.zeros_like(matr)
        filtered_matr = torch.where(matr > 0, matr, filter)

        # Calculate the average values as the expected values
        backup_required_matrix = torch.sum(filtered_matr, dim=0, keepdim=False)
        for i in range(backup_required_matrix.shape[0]):  # Calculate the average
            backup_required_matrix[i] = backup_required_matrix[i] / self.batch_size

        backup_required_matrix_copy = backup_required_matrix.detach()

        # Update Lagrangian multipliers
        if updates % self.Lagrangian_multiplier_update_interval == 0:
            for i in range(self.num_cbfs):
                previous_lambda = self.backup_lambda_values[i]
                new_lambda = previous_lambda + self.augmented_term * backup_required_matrix_copy[i]
                real_new_lambda = torch.clamp(new_lambda, 0.01, 300.0)        # Here set an upper bound for numerical stability
                self.backup_lambda_values[i] = real_new_lambda

        # Update the coefficient of the augmented squared terms
        self.augmented_term = self.augmented_term * self.augmented_ratio
        self.augmented_term = min(self.augmented_term, 200)                    # Here set an upper bound for numerical stability

        backup_policy_loss_2 = float(self.backup_lambda_values[0]) * (
                    backup_required_matrix[0] - self.cost_limit) + self.augmented_term / 2.0 * (
                                    backup_required_matrix[0] - self.cost_limit) * (backup_required_matrix[0] - self.cost_limit)
        for i in range(backup_required_matrix.shape[0] - 1):
            backup_policy_loss_2 += float(self.backup_lambda_values[i + 1]) * (
                        backup_required_matrix[i + 1] - self.cost_limit) + self.augmented_term / 2.0 * (
                                         backup_required_matrix[i + 1] - self.cost_limit) * (
                                         backup_required_matrix[i + 1] - self.cost_limit)

        return backup_policy_loss_2

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : torch.tensor
            min control input.
        u_max : torch.tensor
            max control input.
        """

        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max
