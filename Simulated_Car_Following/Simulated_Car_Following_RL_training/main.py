import argparse
from utils.logx import EpochLogger
import torch
import numpy as np
import random
from sac_cbf_clf.sac_cbf_clf import SAC_CBF_CLF
from sac_cbf_clf.replay_memory import ReplayMemory
from sac_cbf_clf.dynamics import DynamicsModel
from build_env import *
from sac_cbf_clf.utils import prGreen, get_output_folder, prYellow, prCyan
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
    logger.save_config(locals())

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)         # Store the transitions to update the RL-based controller
    NODE_memory = ReplayMemory(args.replay_size, args.seed)    # Store the transitions to update the NODE model of dynamics

    # Training Loop
    total_numsteps = 0
    updates = 0
    start_using_backup = False        # Whether it is permitted to use the backup controller in the current episode
    use_backup = False                # Whether the backup controller is being used at the current timestep


    for i_episode in range(args.max_episodes):
        use_backup = False
        if i_episode >= 0:
            start_using_backup = True   # Only allow the use of the backup controller after several episodes
        backup_time = 0                 # Record the number of timesteps when the backup controller is used. When this value is larger than a pre-defined threshold, stop using the backup controller and reuse the primary controller.
        episode_reward = 0
        episode_cost = 0
        episode_safety_cost = 0.0        # There is an additional term called safety cost and will be used to record the safety cost of each epsiode. Used by CPO, PPO-Lag and TRPO-Lag to form constraints
        episode_steps = 0
        episode_reached = 0              # Record how many times the car is within the required range [9.0,10.0]

        done = False
        obs = env.reset()
        cur_pos_vel_info = np.array([obs[4],obs[5],obs[6],obs[7]])           # Current positions and velocities of 3rd and 4th cars. Here we use this state as the input of the Lyapunov network to decrease the dimension of the input of the Lyapunov network and therefore make the training easier.
        next_pos_vel_info = np.array([obs[4],obs[5],obs[6],obs[7]])          # Next positions and velocities of 3rd and 4th cars. Same with the current information at the beginning of the one episode. Later will be different.

        while not done:
            state = dynamics_model.get_state(obs)  # convert obs to state

            # Train the RL-based controller
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, lyapunov_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates, dynamics_model, NODE_memory, args.NODE_model_update_interval)

                    updates += 1

            # Generate the control signal (action)
            if (use_backup and start_using_backup):
                action = agent.select_action_backup(obs, warmup=args.start_steps > total_numsteps)   #Sample action from the backup policy
                backup_time = backup_time + 1
            else:
                action = agent.select_action(obs, warmup=args.start_steps > total_numsteps)  # Sample action from the primary policy

            next_obs, reward,constraint,cur_pos_vel_info,next_pos_vel_info, done, info = env.step(action)     # Step

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            episode_cost += info.get('num_safety_violation', 0)
            episode_safety_cost += info.get('safety_cost', 0)
            episode_reached += info.get('reached', 0)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            # Append transition to memory
            if not (start_using_backup and use_backup):
                memory.push(obs, action, reward, constraint, cur_pos_vel_info, next_pos_vel_info, next_obs, mask,
                            t=(episode_steps - 1) * env.dt,
                            next_t=(episode_steps) * env.dt)

            # Append transition to NODE_memory
            NODE_memory.push(obs, action, reward, constraint, cur_pos_vel_info, next_pos_vel_info, next_obs, mask,
                             t=episode_steps * env.dt,
                             next_t=(episode_steps + 1) * env.dt)

            next_state = dynamics_model.get_state(next_obs)

            # Decide whether to use the backup controller
            if (start_using_backup and (not use_backup)):
                if (((next_obs[6] * 100.0 - next_obs[8] * 100.0) < 2.5) and (info.get('reached', 0) != 0)):   # When safety constraint is violated while CLF constraint is met successfully, start to use backup controller to achieve and maintain safety.
                    use_backup = True

            if (use_backup and start_using_backup):     # When the time to use the backup controller is longer than the threshold, or the car is far enough from the position where backup controller is started (namely safe enough now), the backup controller will be stopped and the primary controller will be used.
                if backup_time >= 15:
                    use_backup = False
                    backup_time = 0
                if backup_time >= 5 and ((next_obs[4] * 100.0 - next_obs[6] * 100.0) > 2.5) and ((next_obs[6] * 100.0 - next_obs[8] * 100.0) > 2.5):
                    use_backup = False
                    backup_time = 0

            obs = next_obs
            cur_pos_vel_info = next_pos_vel_info

            # env.render(mode='human')       # Choose whether to render.

            if done == True:
                prYellow(
                    'Episode {} - step {} - eps_rew {} - eps_cost {} - eps_safety_cost {}'.format(
                        i_episode, episode_steps, episode_reward, episode_cost,episode_safety_cost))

        # save intermediate model
        if (i_episode % int(args.max_episodes / 6) == 0) or (i_episode == args.max_episodes - 1):
            agent.save_model(args.output)

        # Record the data by using wandb
        writer.log({
            'Episode Reward': episode_reward,
            'Episode Length': episode_steps,
            'Episode Safety Cost': episode_safety_cost,
            'Episode Number of Safety Violations': episode_cost,
            'Cumulated Number of steps': total_numsteps,
            'Episode Number of reaching destination': episode_reached,

        }
        )

        logger.store(Episode=i_episode)
        logger.store(episode_steps=episode_steps)
        logger.store(reward_train=episode_reward)
        logger.store(cost_train=episode_cost)
        logger.store(safety_cost_train=episode_safety_cost)
        logger.store(reached_train=episode_reached)

        # Log the data
        logger.log_tabular('Episode',average_only=True)
        logger.log_tabular('episode_steps',average_only=True)
        logger.log_tabular('reward_train',average_only=True)
        logger.log_tabular('cost_train',average_only=True)
        logger.log_tabular('safety_cost_train', average_only=True)
        logger.log_tabular('reached_train', average_only=True)
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
    parser.add_argument('--NODE_model_update_interval', type=int, default=10, metavar='N',
                        help='NODE model update per no. of updates per step (default: 10)')
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

    exp_name = 'Node_LBAC_SimulatedCarsFollowing'

    # Sets up the output_dir for a logger and returns a dict for logger kwargs
    logger_kwargs = setup_logger_kwargs(exp_name,args.seed,data_dir='./')

    # wandb to save the data and view it on the website
    writer = wandb.init(
        project='Node_LBAC_SimulatedCarsFollowing',  #   Name of the group where data are saved in wandb
        config=args,
        dir='wandb_logs',
        group='SimulatedCars',
    )

    # Train the RL-based controller
    train(agent, env, dynamics_model, args, logger_kwargs=logger_kwargs)

    env.close()

