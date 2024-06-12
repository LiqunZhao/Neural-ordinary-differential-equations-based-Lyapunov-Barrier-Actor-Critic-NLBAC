import argparse
from utils.logx import EpochLogger
import torch
import numpy as np
import random
from sac_cbf_clf.sac_cbf_clf import SAC_CBF_CLF
from sac_cbf_clf.replay_memory import ReplayMemory
from sac_cbf_clf.dynamics import DynamicsModel
from build_env import *
from sac_cbf_clf.utils import prGreen, get_output_folder, prYellow
import wandb


def train(agent, env, dynamics_model, args, logger_kwargs=dict()):
    """
    The RL training process
    :param agent: A SAC_CBF_CLF agent, which includes RL-based controllers
    :param env: A gym environment
    :param dynamics_model: The dynamics model, which helps to convert obs to state
    :param args: Arguments
    :param logger_kwargs: a dict for logger kwargs
    """

    # To log the data
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)         # Store the transitions to update the RL-based controller
    NODE_memory = ReplayMemory(args.replay_size, args.seed)    # Store the transitions to update the NODE model of dynamics

    # Training Loop
    total_numsteps = 0
    updates = 0


    for i_episode in range(args.max_episodes):
        positions_record = []
        episode_reward = 0
        episode_cost = 0
        episode_safety_cost = 0.0
        episode_steps = 0
        done = False
        obs = env.reset()
        cur_cen_pos = np.array([-2.47,-2.5])       # The current center position (used as the input of the Lyapunov network)
        next_center_pos = np.array([-2.47, -2.5])    #Same with cur_cen_pos when initialized.

        while not done:

            state = dynamics_model.get_state(obs)  # convert obs to state

            # Train the RL-based controller
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, lyapunov_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory,args.batch_size,updates,dynamics_model, NODE_memory, args.NODE_model_update_interval)

                    logger.store(critic_1_loss=critic_1_loss)
                    logger.store(critic_2_loss=critic_2_loss)
                    logger.store(lyapunov_loss=lyapunov_loss)
                    logger.store(policy_loss=policy_loss)
                    logger.store(ent_loss=ent_loss)
                    logger.store(alpha=alpha)

                    updates += 1

            # Generate the control signal (action)
            action = agent.select_action(obs,warmup=args.start_steps > total_numsteps)  # Sample action from the primary policy

            next_obs, reward,constraint,barrier_signal,center_pos,next_center_pos, done, info = env.step(action)  # Step

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += info.get('num_safety_violation', 0)
            episode_safety_cost += info.get('safety_cost', 0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            # Append transition to memory
            memory.push(obs, action, reward, constraint, barrier_signal, center_pos, next_center_pos, next_obs, mask,
                        t=episode_steps * env.dt,
                        next_t=(episode_steps + 1) * env.dt)

            # Append transition to NODE_memory
            NODE_memory.push(obs, action, reward, constraint,barrier_signal, center_pos, next_center_pos, next_obs, mask,
                            t=episode_steps * env.dt,
                            next_t=(episode_steps + 1) * env.dt)

            next_state = dynamics_model.get_state(next_obs)

            positions_record.append(next_center_pos)

            if episode_steps >= 50:
                focus_position_record = positions_record[-40:]
                position_difference = focus_position_record[39] - focus_position_record[0]
                x_difference = position_difference[0]
                y_difference = position_difference[1]
                x_difference_square = x_difference * x_difference
                y_difference_square = y_difference * y_difference
                total_difference_square = x_difference_square + y_difference_square

            obs = next_obs
            cur_cen_pos = next_center_pos

            if done == True:
                distance = np.linalg.norm((next_center_pos - [2.5, 2.5]))

        # save intermediate model
        if (i_episode % int(args.max_episodes / 2) == 0) or (i_episode == args.max_episodes - 1):
            agent.save_model(args.output)

        # Record the data by using wandb
        writer.log({
            'Episode Reward': episode_reward,
            'Episode Length': episode_steps,
            'Episode Safety Cost': episode_safety_cost,
            'Episode Number of Safety Violations': episode_cost,
            'Cumulated Number of steps': total_numsteps,

        }
        )


        logger.store(Episode=i_episode)
        logger.store(episode_steps=episode_steps)
        logger.store(reward_train=episode_reward)
        logger.store(cost_train=episode_cost)
        logger.store(safety_cost_train=episode_safety_cost)

        # Log the data
        logger.log_tabular('Episode',average_only=True)
        logger.log_tabular('episode_steps',average_only=True)
        logger.log_tabular('reward_train',average_only=True)
        logger.log_tabular('cost_train',average_only=True)
        logger.log_tabular('safety_cost_train',average_only=True)
        logger.log_tabular('critic_1_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('critic_2_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('lyapunov_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('policy_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('ent_loss',with_min_and_max=True,average_only=False)
        logger.log_tabular('alpha',with_min_and_max=True,average_only=False)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="SimulatedCars", help='Options are Unicycle or SimulatedCars.')
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
    parser.add_argument('--start_steps', type=int, default=3000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--NODE_model_update_interval', type=int, default=10, metavar='N',
                        help='NODE model update per no. of updates per step (default: 10)')
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
    parser.add_argument('--gamma_b', default=20, type=float,help='coefficient used when constructing CBF constraints')
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

    exp_name = 'Node_LBAC_Unicycle'

    # Sets up the output_dir for a logger and returns a dict for logger kwargs
    logger_kwargs = setup_logger_kwargs(exp_name,args.seed,data_dir='./')

    # wandb to save the data and view it on the website
    writer = wandb.init(
        project='Node_neural_barrier_certificate_LBAC_Unicycle',  # Here to change for each different experiment
        config=args,
        dir='wandb_logs',
        group='Unicycle',
    )

    # Train the RL-based controller
    train(agent, env, dynamics_model, args, logger_kwargs=logger_kwargs)

    env.close()
