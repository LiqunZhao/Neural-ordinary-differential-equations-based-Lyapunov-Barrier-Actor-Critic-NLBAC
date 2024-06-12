import argparse
from utils.logx import EpochLogger
import torch
import numpy as np
import random
from sac_cbf_clf.sac_cbf_clf import SAC_CBF_CLF
from sac_cbf_clf.replay_memory import ReplayMemory
from sac_cbf_clf.dynamics import DynamicsModel
from build_env import *
import os
from sac_cbf_clf.utils import prGreen, get_output_folder, prYellow, prPurple
import wandb


def train(agent, env, dynamics_model, args,logger_kwargs=dict()):
    """
    The RL training process
    :param agent: A SAC_CBF_CLF agent, which includes RL-based controllers
    :param env: A gym environment
    :param dynamics_model: The dynamics model, which helps to convert obs to state and state to obs
    :param args: Arguments
    :param logger_kwargs: a dict for logger kwargs
    """

    # To log the data
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)         # Store the transitions to update the RL-based controller
    NODE_memory = ReplayMemory(args.replay_size, args.seed)    # Store the transitions to update the NODE model of dynamics

    # Training Loop
    total_numsteps = 0
    updates = 0
    start_using_backup = False        # Whether it is permitted to use the backup controller in the current episode
    use_backup_obs = False  # Whether to use the backup controller
    use_backup_y = False

    for i_episode in range(args.max_episodes):
        use_backup_obs = False
        use_backup_y = False
        if i_episode >= 3:
            start_using_backup = True   # Only allow the use of the backup controller after several episodes
        positions_record = []
        backup_obs_time = 0  # To record the time to use the backup controller. When the time exceeds the predefined threshold, RL-based controller will be reused.
        backup_y_time = 0
        violation_obs_time = 0
        violation_y_time = 0
        episode_reward = 0
        episode_num_safety_violation_obstacles = 0.0
        episode_num_safety_violation_operator = 0.0
        episode_num_safety_violation_y_min = 0.0
        episode_num_safety_violation_y_max = 0.0
        episode_total_num_safety_violation = 0.0
        episode_safety_cost_obstacles = 0.0
        episode_safety_cost_operator = 0.0
        episode_safety_cost_y_min = 0.0
        episode_safety_cost_y_max = 0.0
        episode_total_safety_cost = 0.0  # There is an additional term called safety cost and will be used to record the safety cost of each epsiode. Used by CPO, PPO-Lag and TRPO-Lag to form constraints
        episode_steps = 0
        done = False
        obs = env.reset()
        lya_pre_term = obs
        lya_next_term = obs

        while not done:

            # Train the RL-based controller
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, lyapunov_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates, dynamics_model, NODE_memory, args.NODE_model_update_interval,i_episode)

                    updates += 1

            # Generate the control signal (action)
            if ((use_backup_obs and start_using_backup) or (use_backup_y and start_using_backup)):
                action = agent.select_action_backup(obs, warmup=args.start_steps > total_numsteps)  # Use the backup controller to have the control signal
                if (use_backup_obs and use_backup_y):
                    backup_obs_time = backup_obs_time + 1
                    backup_y_time = backup_y_time + 1
                elif (use_backup_obs and (not use_backup_y)):
                    backup_obs_time = backup_obs_time + 1
                else:
                    backup_y_time = backup_y_time + 1
            else:
                action = agent.select_action(obs, warmup=args.start_steps > total_numsteps)  # Use the RL-based controller to have the control signal

            next_obs, reward,constraint,lya_pre_term,lya_next_term, done, info = env.step(action)     # Step

            if 'cost_exception' in info:
                prYellow('Cost exception occured.')
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_num_safety_violation_obstacles += info.get('num_safety_violation_obstacles', 0)
            episode_num_safety_violation_operator += info.get('num_safety_violation_safety_operator', 0)
            episode_num_safety_violation_y_min += info.get('num_safety_violation_y_min', 0)
            episode_num_safety_violation_y_max += info.get('num_safety_violation_y_max', 0)
            episode_total_num_safety_violation = episode_num_safety_violation_obstacles + episode_num_safety_violation_operator + episode_num_safety_violation_y_min + episode_num_safety_violation_y_max

            episode_safety_cost_obstacles += info.get('safety_cost_obstacles', 0)
            episode_safety_cost_operator += info.get('safety_cost_operator_val', 0)
            episode_safety_cost_y_min += info.get('safety_cost_y_min_val', 0)
            episode_safety_cost_y_max += info.get('safety_cost_y_max_val', 0)
            episode_total_safety_cost = episode_safety_cost_obstacles + episode_safety_cost_operator + episode_safety_cost_y_min + episode_safety_cost_y_max

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            # Append transition to memory
            if not ((start_using_backup and use_backup_obs) or (use_backup_y and start_using_backup)):
                memory.push(obs, action, reward, constraint, lya_pre_term, lya_next_term, next_obs, mask,
                            t=episode_steps * env.dt,
                            next_t=(episode_steps + 1) * env.dt)

            # Append transition to NODE_memory
            NODE_memory.push(obs, action, reward, constraint, lya_pre_term, lya_next_term, next_obs, mask,
                            t=episode_steps * env.dt,
                            next_t=(episode_steps + 1) * env.dt)


            # Decide whether to use the backup controller

            positions_record.append(lya_next_term)

            if episode_steps >= 50:
                focus_position_record = positions_record[-40:]
                position_difference = focus_position_record[39] - focus_position_record[0]
                x_difference = position_difference[0]
                y_difference = position_difference[1]
                x_difference_square = x_difference * x_difference
                y_difference_square = y_difference * y_difference
                total_difference_square = x_difference_square + y_difference_square

                numpy_focus_position_record = np.array(focus_position_record)
                x_positions_list = numpy_focus_position_record[:, 0]
                y_positions_list = numpy_focus_position_record[:, 1]
                x_positions_list_var = np.var(x_positions_list)
                y_positions_list_var = np.var(y_positions_list)
                position_total_variance = x_positions_list_var + y_positions_list_var

                if (start_using_backup and (
                not use_backup_obs)):  # If total_difference_square is small, it means that the pvtol may get trapped in a specific position, then the backup controller is activated.
                    if total_difference_square <= 0.015:
                        violation_obs_time = violation_obs_time + 1
                        if violation_obs_time >= 8:
                            use_backup_obs = True
                            violation_obs_time = 0
                            x_init_diff = lya_next_term[0]
                            y_init_diff = lya_next_term[1]
                    if (total_difference_square > 0.015) and (violation_obs_time > 0):
                        violation_obs_time = 0

                if (
                        use_backup_obs and start_using_backup):  # When the time to use the backup controller is longer than the threshold, or the pvtol is far enough from the position where backup controller is started (namely far enough from the trapped position), the backup controller will be stopped.
                    if backup_obs_time >= 30:
                        use_backup_obs = False
                        backup_obs_time = 0
                    x_dif = lya_next_term[0] - x_init_diff
                    y_dif = lya_next_term[1] - y_init_diff
                    x_dif_squ = x_dif * x_dif
                    y_dif_squ = y_dif * y_dif
                    tot_dif = x_dif_squ + y_dif_squ
                    if tot_dif >= 1.0:
                        use_backup_obs = False
                        backup_obs_time = 0

                if (start_using_backup and (
                not use_backup_y)):  # When the pvtol is rushing towards the destination but violating the constraint with respect to the safety operator since it rushes to the destination, then the backup controller is activated.
                    if (((lya_next_term[0] <= 4.5) and (lya_next_term[0] - lya_pre_term[0] > 0) and (
                            lya_next_term[0] - lya_next_term[7] > env.operator_dist)) or (
                            (lya_next_term[0] > 4.5) and (lya_next_term[0] - lya_pre_term[0] < 0) and (
                            lya_next_term[7] - lya_next_term[0] > env.operator_dist))):
                        violation_y_time = violation_y_time + 1
                        if violation_y_time >= 1:
                            use_backup_y = True
                            violation_y_time = 0
                    if ((not (((lya_next_term[0] <= 4.5) and (lya_next_term[0] - lya_pre_term[0] > 0) and (
                            lya_next_term[0] - lya_next_term[7] > env.operator_dist)) or (
                                      (lya_next_term[0] > 4.5) and (lya_next_term[0] - lya_pre_term[0] < 0) and (
                                      lya_next_term[7] - lya_next_term[0] > env.operator_dist)))) and (
                            violation_y_time > 0)):
                        violation_y_time = 0

                if (
                        use_backup_y and start_using_backup):  # When the time to use the backup controller is longer than the threshold, or the pvtol does not violate the constraint with respect to the safety operator due to its approach towards the destination, the backup controller will be stopped.
                    if backup_y_time >= 15:
                        use_backup_y = False
                        backup_y_time = 0

                    if (((lya_next_term[0] <= 4.5) and (
                            lya_next_term[0] - lya_next_term[7] <= 0.9 * env.operator_dist)) or (
                            (lya_next_term[0] > 4.5) and (
                            lya_next_term[7] - lya_next_term[0] <= 0.9 * env.operator_dist))):
                        use_backup_y = False
                        backup_y_time = 0

            obs = next_obs
            lya_pre_term = lya_next_term

        final_center_pos = np.array([obs[0], obs[1]])
        final_center_pos_x = obs[0]
        final_center_pos_y = obs[1]
        final_distance = np.linalg.norm((final_center_pos - [4.5, 4.5]))



        # save intermediate model
        if (i_episode % int(args.max_episodes / 2) == 0) or (i_episode == args.max_episodes - 1):
            agent.save_model(args.output)

        # Record the data by using wandb
        writer.log({
            'Episode Reward': episode_reward,
            'Episode Length': episode_steps,
            'Episode Number of Collisions with Obstacles': episode_num_safety_violation_obstacles,
            'Episode Number of Violations concerning Safety Operator': episode_num_safety_violation_operator,
            'Episode Number of Violations concerning ymin': episode_num_safety_violation_y_min,
            'Episode Number of Violations concerning ymax': episode_num_safety_violation_y_max,
            'Episode Number of Safety Violations': episode_total_num_safety_violation,
            'Episode Safety Cost Concerning Obstacles': episode_safety_cost_obstacles,
            'Episode Safety Cost Concerning Safety Operator': episode_safety_cost_operator,
            'Episode Safety Cost Concerning ymin': episode_safety_cost_y_min,
            'Episode Safety Cost Concerning ymax': episode_safety_cost_y_max,
            'Episode Safety Cost': episode_total_safety_cost,
            'Cumulated Number of steps': total_numsteps,

        }
        )

        logger.store(Episode=i_episode)
        logger.store(episode_steps=episode_steps)
        logger.store(reward_train=episode_reward)
        logger.store(cost_train=episode_total_num_safety_violation)
        logger.store(safety_cost_train=episode_total_safety_cost)
        logger.store(final_center_pos_x=final_center_pos_x)
        logger.store(final_center_pos_y=final_center_pos_y)
        logger.store(final_distance=final_distance)


        # Log the data
        logger.log_tabular('Episode', average_only=True)
        logger.log_tabular('episode_steps', average_only=True)
        logger.log_tabular('reward_train', average_only=True)
        logger.log_tabular('cost_train', average_only=True)
        logger.log_tabular('safety_cost_train', average_only=True)
        logger.log_tabular('final_center_pos_x', average_only=True)
        logger.log_tabular('final_center_pos_y', average_only=True)
        logger.log_tabular('final_distance', average_only=True)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="Pvtol", help='Options are Unicycle or SimulatedCars.')
    # SAC Args
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='random seed (default: 12345)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--max_episodes', type=int, default=400, metavar='N',
                        help='maximum number of episodes (default: 400)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--NODE_model_update_interval', type=int, default=10, metavar='N',
                        help='NODE model update per no. of updates per step (default: 10)')
    parser.add_argument('--backup_update_interval', type=int, default=20, metavar='N',
                        help='backup controller update per no. of updates per step (default: 10)')
    parser.add_argument('--start_steps', type=int, default=3000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--Lagrangian_multiplier_update_interval', type=int, default=8, metavar='N',
                        help='Lagrangian_multiplier_update_interval (default: 8)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    # Dynamics, Env Args
    parser.add_argument('--gamma_b', default=0.6, type=float,help='coefficient used when constructing CBF constraints')
    parser.add_argument('--l_p', default=0.03, type=float,
                        help="Look-ahead distance for unicycle dynamics output.")
    args = parser.parse_args()

    # Return save folder
    args.output = get_output_folder(args.output, args.env_name)

    if args.cuda:
        torch.cuda.set_device(args.device_num)


    # Environment and DynamicsModel
    env = build_env(args)
    dynamics_model = DynamicsModel(env, args)

    # Random Seed
    if args.seed >= 0:
        env.seed(args.seed)
        random.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        dynamics_model.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Agent
    agent = SAC_CBF_CLF(env.observation_space.shape[0], env.action_space, env, args)

    from utils.run_utils import setup_logger_kwargs

    exp_name = 'Node_LBAC_Pvtol'

    # Sets up the output_dir for a logger and returns a dict for logger kwargs
    logger_kwargs = setup_logger_kwargs(exp_name,args.seed,data_dir='./')

    # wandb to save the data and view it on the website
    writer = wandb.init(
        project='Node_LBAC_Pvtol',  #   Name of the group where data are saved in wandb
        config=args,
        dir='wandb_logs',
        group='Pvtol',
    )

    # Train the RL-based controller
    train(agent, env, dynamics_model, args, logger_kwargs=logger_kwargs)

    env.close()

