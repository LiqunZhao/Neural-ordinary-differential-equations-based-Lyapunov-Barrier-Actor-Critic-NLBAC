import os
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from envs.unicycle_env import UnicycleEnv

def collect_data_random(env, num_trajectories=20, trajectory_length=100):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    # env.render(mode='human')
    collected_data = []
    for i in tqdm(range(num_trajectories)):
        obs = env.reset()
        state = np.zeros(3)
        next_state = np.zeros(3)
        state[0] = obs[0]
        state[1] = obs[1]
        state[2] = np.arctan2(obs[3], obs[2])
        states = np.zeros((trajectory_length, state.shape[0]), dtype=np.float32)
        next_states = np.zeros((trajectory_length, state.shape[0]), dtype=np.float32)
        actions = np.zeros((trajectory_length, env.action_space.shape[0]), dtype=np.float32)

        for t in range(trajectory_length):

            action = env.action_space.sample()
            states[t] = state
            obs, reward, constraint, center_pos, next_center_pos, done, info = env.step(action)
            actions[t] = action
            next_state[0] = obs[0]
            next_state[1] = obs[1]
            next_state[2] = np.arctan2(obs[3], obs[2])
            next_states[t] = next_state

            state = next_state

        trajectory = {'states': states, 'actions': actions, 'next_states': next_states}
        collected_data.append(trajectory)
        np.save('20_trajectories_100_size_Unicycle_sample_data.npy', collected_data)

    return collected_data

def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:
Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None

    dataset = collected_data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

class NeuralODEModel(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2):
        super(NeuralODEModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        hidden_dim = 100

        # Define the architecture of the model
        self.f_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim1)
        )

        self.g_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim2)
        )


    def forward(self, t, s):
        s_state = s[..., 0:3]
        s_action = s[..., 3:5]
        a = torch.zeros(s_action.shape)
        f_x = self.f_net(s_state)
        g_x = self.g_net(s_state).reshape(-1, 3, 2)
        ds_dt = f_x + torch.bmm(g_x, s_action.reshape(-1, 2, 1)).squeeze()   # Here we assume that we know the dynamics is f(x)+g(x)u.
        ds_dt = torch.cat((ds_dt, a), -1)   # Append ds_dt and a (which is torch.zeros) together to make sure that its dimension (batch_size, 5) aligns with the input.

        return ds_dt



class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    # Define the loss function
    def forward(self, predicted_state, true_state):
        mse_loss = nn.MSELoss(reduction='mean')
        L = mse_loss(predicted_state, true_state)
        return L


def train(train_loader, val_loader, num_epochs, lr, horizon, time_interval, optim, solver='euler'):
    model = NeuralODEModel(3, 3, 6)
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optim == "SGD-M":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if optim == "RMS-Prop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    loss_func = PoseLoss()  # ONLY IF ODE TODO: Needs if condition

    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, loss_func, num_epochs, horizon,
                                           time_interval, solver)

    return model, train_losses, val_losses


def train_model(model, train_dataloader, val_dataloader, optimizer, loss_func, num_epochs, horizon, time_interval,
                solver):
    """
    Trains the given model for `num_epochs` epochs. Use Adam as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param optimizer: Pytorch optimizer.
    :param num_epochs: int, number of epochs to train the model.
    :param horizon: int.
    :param time_interval: float.
    :return:
    """
    train_losses = []
    val_losses = []

    for epoch_i in tqdm(range(num_epochs)):
        train_loss_i = None
        val_loss_i = None


        train_loss_i = train_step(model=model, train_loader=train_dataloader, optimizer=optimizer, loss_func=loss_func,
                                  horizon=horizon, time_interval=time_interval, solver=solver)
        val_loss_i = val_step(model=model, val_loader=val_dataloader, loss_func=loss_func, horizon=horizon,
                              time_interval=time_interval, solver=solver)


        print(
            f"Epoch {epoch_i + 1}/{num_epochs}: train loss={train_loss_i:.4f}, val loss={val_loss_i:.4f}")

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses


def train_step(model, train_loader, optimizer, loss_func, horizon, solver, time_interval) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0.
    t = torch.arange(0, (horizon + 1) * time_interval, time_interval)  # TODO: Make this general based on horizon

    T = horizon
    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        state = batch['states'][:, :T]
        action = batch['actions'][:, :T]
        next_state = batch['next_states'][:, :T]


        # Loop through every action sequence in action and compute the next_state
        next_state_pred = torch.zeros_like(state)
        state_t = state[:,0,:]  # First state always initial state
        for j in range(T):
            action_t = action[:, j, :]
            model_input_t = torch.cat((state_t, action_t), dim=-1)
            t_span = torch.tensor([t[j], t[j + 1]])

            next_state_t = odeint(model, model_input_t, t_span, method=solver, atol=1e-7, rtol=1e-5)[
                -1]  # Predicted value
            next_state_pred[:, j, :] = next_state_t[:, :3]
            if j > 0 and j % 10 == 0:
                state_t = next_state[:, j, :]
            else:
                state_t = next_state_pred[:, j, :]

        loss = loss_func(next_state_pred, next_state)  # COMPARE WITH THE NEXT_STATE


        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def val_step(model, val_loader, loss_func, horizon, solver, time_interval) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0.
    t = torch.arange(0, (horizon + 1) * time_interval, time_interval)  # TODO: Make this general based on horizon
    model.eval()
    T = horizon
    for i, batch in enumerate(val_loader):
        state = batch['states'][:, :T]
        action = batch['actions'][:, :T]
        next_state = batch['next_states'][:, :T]


        # Loop through every action sequence in action and compute the next_state
        next_state_pred = torch.zeros_like(state)
        state_t = state[:, 0, :]  # First state always initial state)
        for j in range(T):
            action_t = action[:, j, :]
            model_input_t = torch.cat((state_t, action_t), dim=-1)
            t_span = torch.tensor([t[j], t[j + 1]])

            next_state_t = odeint(model, model_input_t, t_span, method=solver, atol=1e-7, rtol=1e-5)[
                -1]
            next_state_pred[:, j, :] = next_state_t[:, :3]
            if j > 0 and j % 10 == 0:
                state_t = next_state[:, j, :]
            else:
                state_t = next_state_pred[:, j, :]

        loss = loss_func(next_state_pred, next_state)  # COMPARE WITH THE NEXT_STATE

        val_loss += loss.item()

    return val_loss / len(val_loader)


if __name__ == '__main__':

    env = UnicycleEnv()
    collect_data = collect_data_random(env, num_trajectories=20, trajectory_length=100)
    batch_size = 128
    collected_data = np.load('/home/liqun/Neural-ordinary-differential-equations-based-Lyapunov-Barrier-Actor-Critic-NLBAC/Unicycle/Unicycle_modelling/20_trajectories_100_size_Unicycle_sample_data.npy', allow_pickle=True)
    train_loader, val_loader = process_data_single_step(collected_data, batch_size=batch_size)

    lr = 1e-3
    horizon = 100                   # Equal to the length of each trajectory
    single_horizon = 10             # Use the real state as the initial state for prediction. See Line 235
    solver = 'euler'
    num_epochs = 50
    time_interval = 0.02

    # TRAIN MODEL ##
    trained_model, train_loss, val_loss = train(train_loader, val_loader, num_epochs=num_epochs, lr=lr, horizon=horizon,
                                                optim="Adam", time_interval=time_interval, solver=solver)

    ## SAVE MODEL ##
    name = solver + "_horizon-" + str(horizon) + "_singlehorizon-" + str(single_horizon) + "_" + str(num_epochs) + "_" + str(batch_size) + ".pt"
    model_save_path = os.path.join("trained_models/", name)
    print("Model saved as " + name)
    torch.save(trained_model.state_dict(), model_save_path)
