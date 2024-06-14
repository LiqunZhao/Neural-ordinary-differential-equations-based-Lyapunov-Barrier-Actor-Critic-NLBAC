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
DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2},   # state = [x y θ]
                 'SimulatedCars': {'n_s': 10, 'n_u': 1},  # state = [x y θ v ω]
                 'Pvtol': {'n_s': 5, 'n_u': 2}}  # state = [x y θ v_x v_y (safety_operator)]
MAX_STD = {'Unicycle': [2e-1, 2e-1, 2e-1], 'SimulatedCars': [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2],  'Pvtol': [0, 0, 0, 0, 0]}


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
        self.backup_update_interval = args.backup_update_interval
        self.backup_alpha = args.alpha
        self.lya_pre_term_num = num_inputs   #the number of states as the input of the Lyapunov network as CLF   # How many elements the input of the Lyapunov network used as CLF has

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

        self.lyapunovNet = LyaNetwork(self.lya_pre_term_num,args.hidden_size).to(device=self.device)
        self.lyaNet_optim = Adam(self.lyapunovNet.parameters(),lr = self.critic_lyapunov_lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.lyapunovNet_target = LyaNetwork(self.lya_pre_term_num,args.hidden_size).to(device=self.device)
        hard_update(self.lyapunovNet_target, self.lyapunovNet)

        self.cost_limit = 0.0
        self.augmented_term = 1.0    # initial value of the coefficient of the augmented squared terms
        self.backup_augmented_term = 1.0
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

        elif self.env.dynamics_mode == 'Pvtol':
            self.num_cbfs = (len(env.hazard_locations) + 4)

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
        self.neural_ode_model = NeuralODEModel(6,6,12)
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



    def update_parameters(self, memory, batch_size, updates, dynamics_model, NODE_memory,NODE_model_update_interval,i_episode):
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
        state_batch, action_batch,reward_batch,constraint_batch,lya_pre_term_batch,lya_next_term_batch, next_state_batch, mask_batch, t_batch, next_t_batch = memory.sample(
            batch_size=batch_size)

        # Update the NODE modelling of the dynamics by using the data collected during the RL-based controller training
        if ((i_episode <= 100) and (updates % NODE_model_update_interval == 0)):
            NODE_batch_size = min(NODE_memory.position, 32768)     # Set an upper bound for the number of data used to train the NODE modelling
            node_obs_batch, node_action_batch, node_reward_batch, node_constraint_batch, node_lya_pre_term_batch, node_lya_next_term_batch, node_next_obs_batch, node_mask_batch, node_t_batch, node_next_t_batch = NODE_memory.sample(
                batch_size=NODE_batch_size)
            node_obs_batch = torch.FloatTensor(node_obs_batch).to(self.device)
            node_next_obs_batch = torch.FloatTensor(node_next_obs_batch).to(self.device)
            node_action_batch = torch.FloatTensor(node_action_batch).to(self.device)
            # node_t_batch = torch.FloatTensor(node_t_batch).to(self.device).unsqueeze(1)
            node_state_batch,node_state_dynamics_batch = dynamics_model.get_state(node_obs_batch)
            node_next_state_batch,node_next_state_dynamics_batch = dynamics_model.get_state(node_next_obs_batch)

            average_train_loss = train_step(model=self.neural_ode_model, state=node_state_dynamics_batch,
                                            action=node_action_batch, next_state=node_next_state_dynamics_batch,
                                            optimizer=self.neural_ode_model_optimizer,
                                            loss_func=self.model_loss_func, horizon=NODE_batch_size,
                                            solver=self.solver, time_interval=self.env.dt)
        # The data to train the RL-based controller
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(self.device).unsqueeze(1)
        lya_pre_term_batch = torch.FloatTensor(lya_pre_term_batch).to(self.device)
        # lya_next_term_batch = torch.FloatTensor(lya_next_term_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        # time_batch = torch.FloatTensor(t_batch).to(self.device).unsqueeze(1)
        # next_time_batch = torch.FloatTensor(next_t_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

            lf_next_target = self.lyapunovNet_target(next_state_batch)
            next_l_value = constraint_batch + mask_batch * self.gamma * (lf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        lf = self.lyapunovNet(state_batch)
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



        # The part without CBF or CLF constraints for the primary controller
        policy_loss_1 = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]



        # The CBF and CLF constraints part for the primary controller
        policy_loss_2 = self.get_cbf_clf_part(state_batch, pi, dynamics_model, lya_pre_term_batch,updates)  # Get other terms in the augmented Lagrangian function

        policy_loss = (policy_loss_1 + policy_loss_2)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Do the same things for training the RL-based backup controller if it needs to be updated
        if (updates % self.backup_update_interval == 0):
            backup_pi, backup_log_pi, _ = self.backup_policy.sample(state_batch)

            backup_qf1_pi, backup_qf2_pi = self.critic(state_batch, backup_pi)
            backup_min_qf_pi = torch.min(backup_qf1_pi, backup_qf2_pi)

            # The part without CBF constraints for the backup controller
            backup_policy_loss_1 = ((self.backup_alpha * backup_log_pi) - backup_min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            # The CBF constraints part for the backup controller
            backup_policy_loss_2 = self.backup_get_cbf_clf_part(
                state_batch, backup_pi, dynamics_model,updates)

            backup_policy_loss = (backup_policy_loss_1 + backup_policy_loss_2)

            self.backup_policy_optim.zero_grad()
            backup_policy_loss.backward()
            self.backup_policy_optim.step()

            if self.automatic_entropy_tuning:
                backup_alpha_loss = -(self.backup_log_alpha * (backup_log_pi + self.target_entropy).detach()).mean()

                self.backup_alpha_optim.zero_grad()
                backup_alpha_loss.backward()
                self.backup_alpha_optim.step()

                self.backup_alpha = self.backup_log_alpha.exp()



        if self.automatic_entropy_tuning:     # Update the temperature parameter
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()

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

    def get_cbf_clf_part(self, obs_batch, action_batch, dynamics_model,lya_pre_term_batch,updates):
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

        state_batch,state_dynamics_batch = dynamics_model.get_state(obs_batch)     #convert obs to state
        previous_positions_batch = lya_pre_term_batch.requires_grad_()
        lyapunov_value = self.lyapunovNet(previous_positions_batch)   # Here use previous_positions_batch instead of state_batch as the inputs of the Lyapunov network (since we know the constraint is used to try to achieve a desired distance between the 3rd and 4th cars)
        lyapunov_value_detach = lyapunov_value.detach()
        policy_loss_2 = self.get_policy_loss_2(dynamics_model,state_batch, action_batch,lyapunov_value_detach,updates,state_dynamics_batch)

        return policy_loss_2

    def backup_get_cbf_clf_part(self, obs_batch, action_batch, dynamics_model,updates):   #currently write like this, used when only use QP for finding the one-time action for one-time rollout
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

        state_batch,state_dynamics_batch = dynamics_model.get_state(obs_batch)
        backup_policy_loss_2 = self.backup_get_policy_loss_2(dynamics_model,state_batch, action_batch,updates,state_dynamics_batch)

        return backup_policy_loss_2

    def get_policy_loss_2(self, dynamics_model,state_batch, action_batch,lyapunov_value,updates,state_dynamics_batch):
        assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2, print(state_batch.shape,
                                                                                    action_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b
        gamma_l = 0.1  # define the coefficient for the CLF

        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1)
        state_dynamics_batch = torch.unsqueeze(state_dynamics_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)

        if self.env.dynamics_mode == 'Pvtol':     # Make sure the env is correct
            num_cbfs = self.num_cbfs
            num_clfs = 1
            hazards_radius = self.env.hazards_radius
            hazard_locations = to_tensor(self.env.hazard_locations, torch.FloatTensor, self.device)
            collision_radius = 1.2 * hazards_radius  # add a little buffer


            matrix_seven_to_two = torch.zeros((batch_size, 2, 7)).to(self.device)
            matrix_seven_to_two[:, 0, 0] = 1.0
            matrix_seven_to_two[:, 1, 1] = 1.0

            # Current position y_(t)
            current_pos = torch.bmm(matrix_seven_to_two, state_batch).squeeze(-1)  # batch_size, 2


            state_dynamics_batch_squeeze = state_dynamics_batch.squeeze(-1)
            action_batch_squeeze = action_batch.squeeze(-1)
            model_input_t = torch.cat((state_dynamics_batch_squeeze, action_batch_squeeze), dim=-1).to(device=self.device)
            t_span = torch.tensor([0, self.env.dt]).to(device=self.device)

            # Predict the next state x_(t+1) based on the NODE model
            next_state_pred = odeint(self.neural_ode_model, model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:,:6]

            next_state_pred_unsqueeze = next_state_pred.unsqueeze(-1)

            cur_pos_safety_operator = state_batch[:, 6, :].squeeze(-1)
            next_x_pos = next_state_pred_unsqueeze[:, 0, :].squeeze(-1)
            next_pos_safety_operator = cur_pos_safety_operator + self.env.safety_operator_follow * (
                        next_x_pos - cur_pos_safety_operator)
            next_pos_safety_operator_reshape = torch.reshape(next_pos_safety_operator, (batch_size, 1, 1))

            next_state_for_obs_pred_unsqueeze = torch.cat((next_state_pred_unsqueeze, next_pos_safety_operator_reshape), 1)



            # convert state to obs for sampling the next action u_(t+1)
            next_obs_prediction_batch = dynamics_model.get_obs(next_state_for_obs_pred_unsqueeze,self.device)
            next_obs_prediction_batch_squeeze = next_obs_prediction_batch.squeeze(-1)
            lyapunov_value_next = self.lyapunovNet(next_obs_prediction_batch_squeeze)
            Lya_term = ((lyapunov_value_next - lyapunov_value) / 1.0) + gamma_l * lyapunov_value

            # next position, y_(t+1)
            next_pos = torch.bmm(matrix_seven_to_two, next_state_for_obs_pred_unsqueeze).squeeze(-1)  # batch_size, 2
            #
            # Sample the next action u_(t+1)
            next_obs_prediction_batch_detach = next_obs_prediction_batch.detach()
            next_obs_prediction_batch_detach_squeeze = next_obs_prediction_batch_detach.squeeze(-1)
            pi_next, _, _ = self.policy.sample(next_obs_prediction_batch_detach_squeeze)

            if len(pi_next.shape) == 2:
                pi_next = torch.unsqueeze(pi_next, -1)

            pi_next_detach = pi_next.detach()

            next_state_pred_unsqueeze_squeeze = next_state_pred_unsqueeze.squeeze(-1)
            pi_next_detach_squeeze = pi_next_detach.squeeze(-1)
            next_model_input_t = torch.cat((next_state_pred_unsqueeze_squeeze, pi_next_detach_squeeze), dim=-1).to(
                device=self.device)

            # Predict the next next state x_(t+2) based on the NODE model
            next_next_state_pred = odeint(self.neural_ode_model, next_model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:, :6]

            next_next_state_pred_unsqueeze = next_next_state_pred.unsqueeze(-1)

            next_pos_safety_operator = next_state_for_obs_pred_unsqueeze[:, 6, :].squeeze(-1)
            next_next_x_pos = next_next_state_pred_unsqueeze[:, 0, :].squeeze(-1)
            next_next_pos_safety_operator = next_pos_safety_operator + self.env.safety_operator_follow * (
                    next_next_x_pos - next_pos_safety_operator)
            next_next_pos_safety_operator_reshape = torch.reshape(next_next_pos_safety_operator,
                                                             (batch_size, 1, 1))

            next_next_state_for_obs_pred_unsqueeze = torch.cat((next_next_state_pred_unsqueeze, next_next_pos_safety_operator_reshape),
                                                          1)

            # convert state to obs for sampling the next action u_(t+2)
            next_next_obs_prediction_batch = dynamics_model.get_obs(next_next_state_for_obs_pred_unsqueeze, self.device)
            next_next_obs_prediction_batch_detach = next_next_obs_prediction_batch.detach()
            next_next_obs_prediction_batch_detach_squeeze = next_next_obs_prediction_batch_detach.squeeze(-1)
            pi_next_next, _, _ = self.policy.sample(next_next_obs_prediction_batch_detach_squeeze)

            if len(pi_next_next.shape) == 2:
                pi_next_next = torch.unsqueeze(pi_next_next, -1)

            pi_next_next_detach = pi_next_next.detach()

            # next position, y_(t+2)
            next_next_pos = torch.bmm(matrix_seven_to_two, next_next_state_for_obs_pred_unsqueeze).squeeze(-1)

            next_next_state_pred_unsqueeze_squeeze = next_next_state_pred_unsqueeze.squeeze(-1)
            pi_next_next_detach_squeeze = pi_next_next_detach.squeeze(-1)
            next_next_model_input_t = torch.cat((next_next_state_pred_unsqueeze_squeeze, pi_next_next_detach_squeeze), dim=-1).to(
                device=self.device)

            # Predict the next next state x_(t+2) based on the NODE model
            next_next_next_state_pred = odeint(self.neural_ode_model, next_next_model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:, :6]

            next_next_next_state_pred_unsqueeze = next_next_next_state_pred.unsqueeze(-1)

            next_next_pos_safety_operator = next_next_state_for_obs_pred_unsqueeze[:, 6, :].squeeze(
                -1)
            next_next_next_x_pos = next_next_next_state_pred_unsqueeze[:, 0, :].squeeze(-1)
            next_next_next_pos_safety_operator = next_next_pos_safety_operator + self.env.safety_operator_follow * (
                    next_next_next_x_pos - next_next_pos_safety_operator)
            next_next_next_pos_safety_operator_reshape = torch.reshape(next_next_next_pos_safety_operator,
                                                                  (batch_size, 1, 1))

            next_next_next_state_for_obs_pred_unsqueeze = torch.cat(
                (next_next_next_state_pred_unsqueeze, next_next_next_pos_safety_operator_reshape),
                1)

            # next position, y_(t+3)
            next_next_next_pos = torch.bmm(matrix_seven_to_two, next_next_next_state_for_obs_pred_unsqueeze).squeeze(-1)

            # CBFs h(y_(t))
            ps_hzds = current_pos.repeat((1, len(self.env.hazard_locations))).reshape(
                (batch_size, len(self.env.hazard_locations), 2))
            hs = 0.5 * (torch.sum((ps_hzds - hazard_locations.view(1, len(self.env.hazard_locations), -1)) ** 2,
                                  axis=2) - collision_radius ** 2)

            # CBFs h(y_(t+1))
            ps_next_hzds = next_pos.repeat((1, len(self.env.hazard_locations))).reshape(
                (batch_size, len(self.env.hazard_locations), 2))
            hs_next = 0.5 * (torch.sum(
                (ps_next_hzds - hazard_locations.view(1, len(self.env.hazard_locations), -1)) ** 2,
                axis=2) - collision_radius ** 2)

            # CBFs h(y_(t+2))
            ps_next_next_hzds = next_next_pos.repeat((1, len(self.env.hazard_locations))).reshape(
                (batch_size, len(self.env.hazard_locations), 2))
            hs_next_next = 0.5 * (torch.sum(
                (ps_next_next_hzds - hazard_locations.view(1, len(self.env.hazard_locations), -1)) ** 2,
                axis=2) - collision_radius ** 2)

            # CBFs h(y_(t+3))
            ps_next_next_next_hzds = next_next_next_pos.repeat((1, len(self.env.hazard_locations))).reshape(
                (batch_size, len(self.env.hazard_locations), 2))
            hs_next_next_next = 0.5 * (torch.sum(
                (ps_next_next_next_hzds - hazard_locations.view(1, len(self.env.hazard_locations), -1)) ** 2,
                axis=2) - collision_radius ** 2)

            cbf_obstacles_term1 = hs_next_next_next - hs_next_next + gamma_b * hs_next_next
            cbf_obstacles_term2 = hs_next_next - hs_next + gamma_b * hs_next
            cbf_obstacles_term3 = hs_next - hs + gamma_b * hs

            cbf_term_obstacles = - (cbf_obstacles_term1 - cbf_obstacles_term2 + gamma_b * cbf_obstacles_term2 - (
                        cbf_obstacles_term2 - cbf_obstacles_term3 + gamma_b * cbf_obstacles_term3) + gamma_b * (
                                                cbf_obstacles_term2 - cbf_obstacles_term3 + gamma_b * cbf_obstacles_term3))


            # 1st constraint considering the safety operator
            operator_dist = 0.9 * self.env.operator_dist
            matrix_seven_to_one = torch.zeros((batch_size, 1, 7)).to(self.device)
            matrix_seven_to_one[:, 0, 0] = 1.0
            matrix_seven_to_one[:, 0, 6] = -1.0
            h1 = (torch.bmm(matrix_seven_to_one, state_batch) + operator_dist).squeeze(-1)
            h1_next = (torch.bmm(matrix_seven_to_one, next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(-1)
            h1_next_next = (torch.bmm(matrix_seven_to_one, next_next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(
                -1)
            h1_next_next_next = (
                    torch.bmm(matrix_seven_to_one, next_next_next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(
                -1)

            cbf_h1_term1 = h1_next_next_next - h1_next_next + gamma_b * h1_next_next
            cbf_h1_term2 = h1_next_next - h1_next + gamma_b * h1_next
            cbf_h1_term3 = h1_next - h1 + gamma_b * h1

            cbf_term_operator_1 = - (cbf_h1_term1 - cbf_h1_term2 + gamma_b * cbf_h1_term2 - (
                        cbf_h1_term2 - cbf_h1_term3 + gamma_b * cbf_h1_term3) + gamma_b * (
                                                 cbf_h1_term2 - cbf_h1_term3 + gamma_b * cbf_h1_term3))


            # 2nd constraint considering the safety operator
            matrix_seven_to_one_2 = torch.zeros((batch_size, 1, 7)).to(self.device)
            matrix_seven_to_one_2[:, 0, 0] = -1.0
            matrix_seven_to_one_2[:, 0, 6] = 1.0
            h2 = (torch.bmm(matrix_seven_to_one_2, state_batch) + operator_dist).squeeze(-1)
            h2_next = (torch.bmm(matrix_seven_to_one_2, next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(-1)
            h2_next_next = (
                    torch.bmm(matrix_seven_to_one_2, next_next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(
                -1)
            h2_next_next_next = (
                    torch.bmm(matrix_seven_to_one_2, next_next_next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(
                -1)

            cbf_h2_term1 = h2_next_next_next - h2_next_next + gamma_b * h2_next_next
            cbf_h2_term2 = h2_next_next - h2_next + gamma_b * h2_next
            cbf_h2_term3 = h2_next - h2 + gamma_b * h2

            cbf_term_operator_2 = - (cbf_h2_term1 - cbf_h2_term2 + gamma_b * cbf_h2_term2 - (
                        cbf_h2_term2 - cbf_h2_term3 + gamma_b * cbf_h2_term3) + gamma_b * (
                                                 cbf_h2_term2 - cbf_h2_term3 + gamma_b * cbf_h2_term3))

            # h3, ymax
            delta_y = 10.0
            matrix_seven_to_one_y = torch.zeros((batch_size, 1, 7)).to(self.device)
            matrix_seven_to_one_y[:, 0, 1] = -1.0
            h3 = (torch.bmm(matrix_seven_to_one_y, state_batch) + self.env.y_max - delta_y).squeeze(-1)
            h3_next = (torch.bmm(matrix_seven_to_one_y,
                                 next_state_for_obs_pred_unsqueeze) + self.env.y_max - delta_y).squeeze(-1)
            h3_next_next = (torch.bmm(matrix_seven_to_one_y,
                                      next_next_state_for_obs_pred_unsqueeze) + self.env.y_max - delta_y).squeeze(-1)
            h3_next_next_next = (torch.bmm(matrix_seven_to_one_y,
                                           next_next_next_state_for_obs_pred_unsqueeze) + self.env.y_max - delta_y).squeeze(
                -1)

            cbf_y_max_term1 = h3_next_next_next - h3_next_next + gamma_b * h3_next_next
            cbf_y_max_term2 = h3_next_next - h3_next + gamma_b * h3_next
            cbf_y_max_term3 = h3_next - h3 + gamma_b * h3

            cbf_y_max = - (cbf_y_max_term1 - cbf_y_max_term2 + gamma_b * cbf_y_max_term2 - (
                        cbf_y_max_term2 - cbf_y_max_term3 + gamma_b * cbf_y_max_term3) + gamma_b * (
                                       cbf_y_max_term2 - cbf_y_max_term3 + gamma_b * cbf_y_max_term3))


            # h4, ymin
            matrix_seven_to_one_y_two = torch.zeros((batch_size, 1, 7)).to(self.device)
            matrix_seven_to_one_y_two[:, 0, 1] = 1.0
            h4 = (torch.bmm(matrix_seven_to_one_y_two, state_batch) - self.env.y_min - delta_y).squeeze(-1)
            h4_next = (torch.bmm(matrix_seven_to_one_y_two,
                                 next_state_for_obs_pred_unsqueeze) - self.env.y_min - delta_y).squeeze(-1)
            h4_next_next = (torch.bmm(matrix_seven_to_one_y_two,
                                      next_next_state_for_obs_pred_unsqueeze) - self.env.y_min - delta_y).squeeze(-1)
            h4_next_next_next = (torch.bmm(matrix_seven_to_one_y_two,
                                           next_next_next_state_for_obs_pred_unsqueeze) - self.env.y_min - delta_y).squeeze(
                -1)

            cbf_y_min_term1 = h4_next_next_next - h4_next_next + gamma_b * h4_next_next
            cbf_y_min_term2 = h4_next_next - h4_next + gamma_b * h4_next
            cbf_y_min_term3 = h4_next - h4 + gamma_b * h4

            cbf_y_min = - (cbf_y_min_term1 - cbf_y_min_term2 + gamma_b * cbf_y_min_term2 - (
                        cbf_y_min_term2 - cbf_y_min_term3 + gamma_b * cbf_y_min_term3) + gamma_b * (
                                       cbf_y_min_term2 - cbf_y_min_term3 + gamma_b * cbf_y_min_term3))


            n_u = action_batch.shape[1]  # dimension of control inputs
            self.num_u = n_u

            ineq_constraint_counter = 0

            self.num_x = n_u

            ineq_constraint_counter += (num_cbfs + num_clfs)


        else:
            raise Exception('Dynamics mode unknown!')

        # Combine the CBF and CLF constraints within one matrix
        matr = torch.cat((cbf_term_obstacles,cbf_term_operator_1,cbf_term_operator_2,cbf_y_max,cbf_y_min,Lya_term),1)  # Obtain the matrix containing all the CBF and CLF information
        matr = matr.unsqueeze(-1)

        # Filter the matrix and only leave the ones whose values are larger than 0 (all the elements whose values are smaller than 0, which means CBF and CLF constraints have been satisfied at those states, are abandoned and replaced by 0 without gradient information). Similar to Relu function.
        filter = torch.zeros_like(matr)
        filtered_matr = torch.where(matr > 0, matr, filter)

        # Calculate the average values as the expected values
        required_matrix = torch.sum(filtered_matr, dim=0, keepdim=False)
        for i in range(required_matrix.shape[0]):  # Calculate the average
            required_matrix[i] = required_matrix[i] / self.batch_size

        # Here calculate a ratio that is used to try to keep a balance between the CBF and CLF constraint losses
        other_compoenent = required_matrix[:-1, :] - self.cost_limit
        other_compoenent_mean = torch.abs(torch.mean(other_compoenent))
        lya_component = torch.abs(required_matrix[-1, :] - self.cost_limit)
        ratio = float(other_compoenent_mean / lya_component)

        if ratio < 0.002:  # Even want to keep the balance, the ratio cannot be too small.
            ratio = 0.002

        required_matrix_copy = required_matrix.detach()

        # Update Lagrangian multipliers
        if updates % self.Lagrangian_multiplier_update_interval == 0:
            for i in range(self.num_constraints):
                previous_lambda = self.lambda_values[i]
                new_lambda = previous_lambda + self.augmented_term * required_matrix_copy[i]
                real_new_lambda = torch.clamp(new_lambda, 0.01, 400.0)
                self.lambda_values[i] = real_new_lambda

        # Update the coefficient of the augmented squared terms
        self.augmented_term = self.augmented_term * self.augmented_ratio  # Enlarge the coefficient for quadratic term. Also confine it to be within a range.
        self.augmented_term = min(self.augmented_term, 200)

        policy_loss_2 = float(self.lambda_values[0]) * (
                required_matrix[0] - self.cost_limit) + self.augmented_term / 2.0 * (
                                required_matrix[0] - self.cost_limit) * (required_matrix[0] - self.cost_limit)
        for i in range(required_matrix.shape[0] - 2):
            policy_loss_2 += float(self.lambda_values[i + 1]) * (
                    required_matrix[i + 1] - self.cost_limit) + self.augmented_term / 2.0 * (
                                     required_matrix[i + 1] - self.cost_limit) * (
                                     required_matrix[i + 1] - self.cost_limit)
        policy_loss_2 += float(self.lambda_values[-1]) * ratio * (
                required_matrix[-1] - self.cost_limit) + ratio * ratio * self.augmented_term / 2.0 * (
                                 required_matrix[-1] - self.cost_limit) * (
                                 required_matrix[-1] - self.cost_limit)

        return policy_loss_2

    def backup_get_policy_loss_2(self, dynamics_model, state_batch, action_batch, updates, state_dynamics_batch):
        assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2, print(state_batch.shape,
                                                                                   action_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b

        # Expand dims
        state_batch = torch.unsqueeze(state_batch, -1)
        state_dynamics_batch = torch.unsqueeze(state_dynamics_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)

        if self.env.dynamics_mode == 'Pvtol':  # Make sure the env is correct
            num_cbfs = self.num_cbfs
            num_clfs = 1
            hazards_radius = self.env.hazards_radius
            hazard_locations = to_tensor(self.env.hazard_locations, torch.FloatTensor, self.device)
            collision_radius = 1.2 * hazards_radius  # add a little buffer

            matrix_seven_to_two = torch.zeros((batch_size, 2, 7)).to(self.device)
            matrix_seven_to_two[:, 0, 0] = 1.0
            matrix_seven_to_two[:, 1, 1] = 1.0

            # Current position y_(t)
            current_pos = torch.bmm(matrix_seven_to_two, state_batch).squeeze(-1)

            state_dynamics_batch_squeeze = state_dynamics_batch.squeeze(-1)
            action_batch_squeeze = action_batch.squeeze(-1)
            model_input_t = torch.cat((state_dynamics_batch_squeeze, action_batch_squeeze), dim=-1).to(
                device=self.device)
            t_span = torch.tensor([0, self.env.dt]).to(device=self.device)

            # Predict the next state x_(t+1) based on the NODE model
            next_state_pred = \
            odeint(self.neural_ode_model, model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:, :6]

            next_state_pred_unsqueeze = next_state_pred.unsqueeze(-1)

            cur_pos_safety_operator = state_batch[:, 6, :].squeeze(-1)
            next_x_pos = next_state_pred_unsqueeze[:, 0, :].squeeze(-1)
            next_pos_safety_operator = cur_pos_safety_operator + self.env.safety_operator_follow * (
                    next_x_pos - cur_pos_safety_operator)
            next_pos_safety_operator_reshape = torch.reshape(next_pos_safety_operator,
                                                             (batch_size, 1, 1))

            next_state_for_obs_pred_unsqueeze = torch.cat((next_state_pred_unsqueeze, next_pos_safety_operator_reshape),
                                                          1)

            # convert state to obs for sampling the next action u_(t+1)
            next_obs_prediction_batch = dynamics_model.get_obs(next_state_for_obs_pred_unsqueeze, self.device)

            # next position, y_(t+1)
            next_pos = torch.bmm(matrix_seven_to_two, next_state_for_obs_pred_unsqueeze).squeeze(-1)  # batch_size, 2

            # Sample the next action u_(t+1)
            next_obs_prediction_batch_detach = next_obs_prediction_batch.detach()
            next_obs_prediction_batch_detach_squeeze = next_obs_prediction_batch_detach.squeeze(-1)
            pi_next, _, _ = self.backup_policy.sample(next_obs_prediction_batch_detach_squeeze)

            if len(pi_next.shape) == 2:
                pi_next = torch.unsqueeze(pi_next, -1)

            pi_next_detach = pi_next.detach()

            next_state_pred_unsqueeze_squeeze = next_state_pred_unsqueeze.squeeze(-1)
            pi_next_detach_squeeze = pi_next_detach.squeeze(-1)
            next_model_input_t = torch.cat((next_state_pred_unsqueeze_squeeze, pi_next_detach_squeeze), dim=-1).to(
                device=self.device)

            # Predict the next next state x_(t+2) based on the NODE model
            next_next_state_pred = \
            odeint(self.neural_ode_model, next_model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:, :6]

            next_next_state_pred_unsqueeze = next_next_state_pred.unsqueeze(-1)

            next_pos_safety_operator = next_state_for_obs_pred_unsqueeze[:, 6, :].squeeze(
                -1)
            next_next_x_pos = next_next_state_pred_unsqueeze[:, 0, :].squeeze(-1)
            next_next_pos_safety_operator = next_pos_safety_operator + self.env.safety_operator_follow * (
                    next_next_x_pos - next_pos_safety_operator)
            next_next_pos_safety_operator_reshape = torch.reshape(next_next_pos_safety_operator,
                                                                  (batch_size, 1, 1))

            next_next_state_for_obs_pred_unsqueeze = torch.cat(
                (next_next_state_pred_unsqueeze, next_next_pos_safety_operator_reshape),
                1)

            # convert state to obs for sampling the next action u_(t+2)
            next_next_obs_prediction_batch = dynamics_model.get_obs(next_next_state_for_obs_pred_unsqueeze, self.device)
            next_next_obs_prediction_batch_detach = next_next_obs_prediction_batch.detach()
            next_next_obs_prediction_batch_detach_squeeze = next_next_obs_prediction_batch_detach.squeeze(-1)
            pi_next_next, _, _ = self.backup_policy.sample(next_next_obs_prediction_batch_detach_squeeze)

            if len(pi_next_next.shape) == 2:
                pi_next_next = torch.unsqueeze(pi_next_next, -1)

            pi_next_next_detach = pi_next_next.detach()

            # next position, y_(t+2)
            next_next_pos = torch.bmm(matrix_seven_to_two, next_next_state_for_obs_pred_unsqueeze).squeeze(
                -1)

            next_next_state_pred_unsqueeze_squeeze = next_next_state_pred_unsqueeze.squeeze(-1)
            pi_next_next_detach_squeeze = pi_next_next_detach.squeeze(-1)
            next_next_model_input_t = torch.cat((next_next_state_pred_unsqueeze_squeeze, pi_next_next_detach_squeeze),
                                                dim=-1).to(
                device=self.device)

            # Predict the next next state x_(t+2) based on the NODE model
            next_next_next_state_pred = \
            odeint(self.neural_ode_model, next_next_model_input_t, t_span, method=self.solver, atol=1e-7, rtol=1e-5)[
                -1][:, :6]

            next_next_next_state_pred_unsqueeze = next_next_next_state_pred.unsqueeze(-1)

            next_next_pos_safety_operator = next_next_state_for_obs_pred_unsqueeze[:, 6, :].squeeze(
                -1)
            next_next_next_x_pos = next_next_next_state_pred_unsqueeze[:, 0, :].squeeze(-1)
            next_next_next_pos_safety_operator = next_next_pos_safety_operator + self.env.safety_operator_follow * (
                    next_next_next_x_pos - next_next_pos_safety_operator)
            next_next_next_pos_safety_operator_reshape = torch.reshape(next_next_next_pos_safety_operator,
                                                                       (batch_size, 1, 1))

            next_next_next_state_for_obs_pred_unsqueeze = torch.cat(
                (next_next_next_state_pred_unsqueeze, next_next_next_pos_safety_operator_reshape),
                1)

            # next position, y_(t+3)
            next_next_next_pos = torch.bmm(matrix_seven_to_two, next_next_next_state_for_obs_pred_unsqueeze).squeeze(
                -1)

            # CBFs h(y_(t))
            ps_hzds = current_pos.repeat((1, len(self.env.hazard_locations))).reshape(
                (batch_size, len(self.env.hazard_locations), 2))
            hs = 0.5 * (torch.sum((ps_hzds - hazard_locations.view(1, len(self.env.hazard_locations), -1)) ** 2,
                                  axis=2) - collision_radius ** 2)  # 1/2 * (||x - x_obs||^2 - r^2)

            # CBFs h(y_(t+1))
            ps_next_hzds = next_pos.repeat((1, len(self.env.hazard_locations))).reshape(
                (batch_size, len(self.env.hazard_locations), 2))
            hs_next = 0.5 * (torch.sum(
                (ps_next_hzds - hazard_locations.view(1, len(self.env.hazard_locations), -1)) ** 2,
                axis=2) - collision_radius ** 2)

            # CBFs h(y_(t+2))
            ps_next_next_hzds = next_next_pos.repeat((1, len(self.env.hazard_locations))).reshape(
                (batch_size, len(self.env.hazard_locations), 2))
            hs_next_next = 0.5 * (torch.sum(
                (ps_next_next_hzds - hazard_locations.view(1, len(self.env.hazard_locations), -1)) ** 2,
                axis=2) - collision_radius ** 2)

            # CBFs h(y_(t+3))
            ps_next_next_next_hzds = next_next_next_pos.repeat((1, len(self.env.hazard_locations))).reshape(
                (batch_size, len(self.env.hazard_locations), 2))
            hs_next_next_next = 0.5 * (torch.sum(
                (ps_next_next_next_hzds - hazard_locations.view(1, len(self.env.hazard_locations), -1)) ** 2,
                axis=2) - collision_radius ** 2)

            cbf_obstacles_term1 = hs_next_next_next - hs_next_next + gamma_b * hs_next_next
            cbf_obstacles_term2 = hs_next_next - hs_next + gamma_b * hs_next
            cbf_obstacles_term3 = hs_next - hs + gamma_b * hs

            cbf_term_obstacles = - (cbf_obstacles_term1 - cbf_obstacles_term2 + gamma_b * cbf_obstacles_term2 - (
                    cbf_obstacles_term2 - cbf_obstacles_term3 + gamma_b * cbf_obstacles_term3) + gamma_b * (
                                            cbf_obstacles_term2 - cbf_obstacles_term3 + gamma_b * cbf_obstacles_term3))


            # 1st constraint considering the safety operator
            operator_dist = 0.9 * self.env.operator_dist
            matrix_seven_to_one = torch.zeros((batch_size, 1, 7)).to(self.device)
            matrix_seven_to_one[:, 0, 0] = 1.0
            matrix_seven_to_one[:, 0, 6] = -1.0
            h1 = (torch.bmm(matrix_seven_to_one, state_batch) + operator_dist).squeeze(-1)
            h1_next = (torch.bmm(matrix_seven_to_one, next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(-1)
            h1_next_next = (
                        torch.bmm(matrix_seven_to_one, next_next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(
                -1)
            h1_next_next_next = (
                    torch.bmm(matrix_seven_to_one,
                              next_next_next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(
                -1)

            cbf_h1_term1 = h1_next_next_next - h1_next_next + gamma_b * h1_next_next
            cbf_h1_term2 = h1_next_next - h1_next + gamma_b * h1_next
            cbf_h1_term3 = h1_next - h1 + gamma_b * h1

            cbf_term_operator_1 = - (cbf_h1_term1 - cbf_h1_term2 + gamma_b * cbf_h1_term2 - (
                    cbf_h1_term2 - cbf_h1_term3 + gamma_b * cbf_h1_term3) + gamma_b * (
                                             cbf_h1_term2 - cbf_h1_term3 + gamma_b * cbf_h1_term3))

            # 2nd constraint considering the safety operator
            matrix_seven_to_one_2 = torch.zeros((batch_size, 1, 7)).to(self.device)
            matrix_seven_to_one_2[:, 0, 0] = -1.0
            matrix_seven_to_one_2[:, 0, 6] = 1.0
            h2 = (torch.bmm(matrix_seven_to_one_2, state_batch) + operator_dist).squeeze(-1)
            h2_next = (torch.bmm(matrix_seven_to_one_2, next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(-1)
            h2_next_next = (
                    torch.bmm(matrix_seven_to_one_2, next_next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(
                -1)
            h2_next_next_next = (
                    torch.bmm(matrix_seven_to_one_2,
                              next_next_next_state_for_obs_pred_unsqueeze) + operator_dist).squeeze(
                -1)

            cbf_h2_term1 = h2_next_next_next - h2_next_next + gamma_b * h2_next_next
            cbf_h2_term2 = h2_next_next - h2_next + gamma_b * h2_next
            cbf_h2_term3 = h2_next - h2 + gamma_b * h2

            cbf_term_operator_2 = - (cbf_h2_term1 - cbf_h2_term2 + gamma_b * cbf_h2_term2 - (
                    cbf_h2_term2 - cbf_h2_term3 + gamma_b * cbf_h2_term3) + gamma_b * (
                                             cbf_h2_term2 - cbf_h2_term3 + gamma_b * cbf_h2_term3))

            # h3, ymax
            delta_y = 10.0
            matrix_seven_to_one_y = torch.zeros((batch_size, 1, 7)).to(self.device)
            matrix_seven_to_one_y[:, 0, 1] = -1.0
            h3 = (torch.bmm(matrix_seven_to_one_y, state_batch) + self.env.y_max - delta_y).squeeze(-1)
            h3_next = (torch.bmm(matrix_seven_to_one_y,
                                 next_state_for_obs_pred_unsqueeze) + self.env.y_max - delta_y).squeeze(-1)
            h3_next_next = (torch.bmm(matrix_seven_to_one_y,
                                      next_next_state_for_obs_pred_unsqueeze) + self.env.y_max - delta_y).squeeze(-1)
            h3_next_next_next = (torch.bmm(matrix_seven_to_one_y,
                                           next_next_next_state_for_obs_pred_unsqueeze) + self.env.y_max - delta_y).squeeze(
                -1)

            cbf_y_max_term1 = h3_next_next_next - h3_next_next + gamma_b * h3_next_next
            cbf_y_max_term2 = h3_next_next - h3_next + gamma_b * h3_next
            cbf_y_max_term3 = h3_next - h3 + gamma_b * h3

            cbf_y_max = - (cbf_y_max_term1 - cbf_y_max_term2 + gamma_b * cbf_y_max_term2 - (
                    cbf_y_max_term2 - cbf_y_max_term3 + gamma_b * cbf_y_max_term3) + gamma_b * (
                                   cbf_y_max_term2 - cbf_y_max_term3 + gamma_b * cbf_y_max_term3))

            # h4, ymin
            matrix_seven_to_one_y_two = torch.zeros((batch_size, 1, 7)).to(self.device)
            matrix_seven_to_one_y_two[:, 0, 1] = 1.0
            h4 = (torch.bmm(matrix_seven_to_one_y_two, state_batch) - self.env.y_min - delta_y).squeeze(-1)
            h4_next = (torch.bmm(matrix_seven_to_one_y_two,
                                 next_state_for_obs_pred_unsqueeze) - self.env.y_min - delta_y).squeeze(-1)
            h4_next_next = (torch.bmm(matrix_seven_to_one_y_two,
                                      next_next_state_for_obs_pred_unsqueeze) - self.env.y_min - delta_y).squeeze(-1)
            h4_next_next_next = (torch.bmm(matrix_seven_to_one_y_two,
                                           next_next_next_state_for_obs_pred_unsqueeze) - self.env.y_min - delta_y).squeeze(
                -1)

            cbf_y_min_term1 = h4_next_next_next - h4_next_next + gamma_b * h4_next_next
            cbf_y_min_term2 = h4_next_next - h4_next + gamma_b * h4_next
            cbf_y_min_term3 = h4_next - h4 + gamma_b * h4

            cbf_y_min = - (cbf_y_min_term1 - cbf_y_min_term2 + gamma_b * cbf_y_min_term2 - (
                    cbf_y_min_term2 - cbf_y_min_term3 + gamma_b * cbf_y_min_term3) + gamma_b * (
                                   cbf_y_min_term2 - cbf_y_min_term3 + gamma_b * cbf_y_min_term3))

            n_u = action_batch.shape[1]  # dimension of control inputs
            self.num_u = n_u

            ineq_constraint_counter = 0

            self.num_x = n_u

            ineq_constraint_counter += (num_cbfs + num_clfs)


        else:
            raise Exception('Dynamics mode unknown!')

        # Combine the CBF constraints within one matrix
        matr = torch.cat((cbf_term_obstacles, cbf_term_operator_1, cbf_term_operator_2, cbf_y_max, cbf_y_min),
                         1)  # Obtain the matrix containing all the CBF information
        matr = matr.unsqueeze(-1)

        # Filter the matrix and only leave the ones whose values are larger than 0 (all the elements whose values are smaller than 0, which means CBF and CLF constraints have been satisfied at those states, are abandoned and replaced by 0 without gradient information). Similar to Relu function.
        filter = torch.zeros_like(matr)
        filtered_matr = torch.where(matr > 0, matr, filter)

        # Calculate the average values as the expected values
        backup_required_matrix = torch.sum(filtered_matr, dim=0, keepdim=False)
        for i in range(backup_required_matrix.shape[0]):
            backup_required_matrix[i] = backup_required_matrix[i] / self.batch_size

        backup_required_matrix_copy = backup_required_matrix.detach()

        # Update Lagrangian multipliers
        if updates % (self.Lagrangian_multiplier_update_interval * self.backup_update_interval) == 0:
            for i in range(self.num_cbfs):
                previous_lambda = self.backup_lambda_values[i]
                new_lambda = previous_lambda + self.backup_augmented_term * backup_required_matrix_copy[i]
                real_new_lambda = torch.clamp(new_lambda, 0.01, 400.0)
                self.backup_lambda_values[i] = real_new_lambda

        # Update the coefficient of the augmented squared terms
        self.backup_augmented_term = self.backup_augmented_term * self.augmented_ratio  # Enlarge the coefficient for quadratic term. Also confine it to be within a range.
        self.backup_augmented_term = min(self.backup_augmented_term, 200)

        backup_policy_loss_2 = float(self.backup_lambda_values[0]) * (
                backup_required_matrix[0] - self.cost_limit) + self.backup_augmented_term / 2.0 * (
                                backup_required_matrix[0] - self.cost_limit) * (backup_required_matrix[0] - self.cost_limit)
        for i in range(backup_required_matrix.shape[0] - 1):
            backup_policy_loss_2 += float(self.backup_lambda_values[i + 1]) * (
                    backup_required_matrix[i + 1] - self.cost_limit) + self.backup_augmented_term / 2.0 * (
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