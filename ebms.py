import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from dynamics import linear_dynamics

class Args:
    batch_size = 64
    learning_rate = 0.005
    optimizer = 'Adam'
    num_layers = 3
    layer_dims = (32, 64, 1)
    dt = 0.01
    num_epochs = 1000
    negative_sample_variance = 0.1
    lr_scheduler_step_size = 50
    lr_scheduler_gamma = 0.995
    dynamics_function = linear_dynamics
    state_dim = 2  # Define the dimension of the state here
    margin = 10
    seed = 4 #TODO: add seed in random

class EnergyPredictor(nn.Module):
    def __init__(self, args):
        super(EnergyPredictor, self).__init__()
        layers = [
            nn.Linear(args.state_dim, args.layer_dims[0]),
            nn.ReLU()
        ]
        for i in range(args.num_layers - 1):
            layers.append(nn.Linear(args.layer_dims[i], args.layer_dims[i + 1]))
            if i < args.num_layers - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def contrastive_loss(model, x_t, x_tp1, x_neg, margin=Args.margin):
    """Computes the contrastive loss.

    Args:
        model (nn.Module): The neural network model.
        x_t (torch.Tensor): The current state tensor.
        x_tp1 (torch.Tensor): The next state tensor.
        x_neg (torch.Tensor): The negative samples tensor.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    E_xt = model(x_t)
    
    #E_xtp1 = model(x_tp1)
    E_xneg = model(x_neg)
    loss = torch.mean(E_xt ** 2) + torch.relu(margin - torch.mean(E_xneg ** 2))
    return loss

def train(model, args, dynamics_fn, loss_fn):
    """Trains the model on the given dataset.

    Args:
        model (nn.Module): The neural network model.
        args (Args): The command line arguments.
        dynamics_fn (function): The dynamics function.
        loss_fn (function): The loss function.
    """
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Invalid optimizer specified in Args.")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma)

    for epoch in tqdm(range(args.num_epochs)):
        x_t = torch.tensor(np.random.random((args.batch_size, 2)), dtype=torch.float32)
        x_tp1 = dynamics_fn(x_t, args.dt)
        noise = np.random.normal(0, args.negative_sample_variance, (args.batch_size, 2))
        x_neg = torch.tensor(x_t.numpy() + noise, dtype=torch.float32)

        optimizer.zero_grad()
        loss = loss_fn(model, x_t, x_tp1, x_neg)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def evaluate(model, x_t, dt, dynamics_fn, num_steps):
    """Evaluates the model on given dataset.

    Args:
        model (nn.Module): The neural network model.
        x_t (torch.Tensor): The current state tensor.
        dt (float): The time step.
        dynamics_fn (function): The dynamics function.
        num_steps (int): The number of evaluation steps.

    Returns:
        np.ndarray: Array of prediction errors.
    """
    errors = []

    for _ in range(1,num_steps):
        x_t.requires_grad_(True)  # Ensure that x_t has requires_grad=True
        E_xt = model(x_t)
        E_xt.backward(torch.ones_like(E_xt))
        grad_E_xt = x_t.grad
        x_t_inferred = x_t + dt * grad_E_xt
        x_t_actual = dynamics_fn(x_t.detach(), dt)
        error = torch.norm(x_t_inferred.detach() - x_t_actual, dim=-1).numpy()
        errors.append(error)
        x_t = x_t_actual

    return np.stack(errors)

def plot_energy(model, x_ranges, axes=None):
    """
    Plots the energy landscape of the model. If the system has one dimension, a 2D plot is
    generated. For systems with more than one dimension, a 3D plot is created, with the option to
    specify which axes to plot.

    Args:
        model (nn.Module): The neural network model.
        x_ranges (list of np.ndarray): A list of 1D NumPy arrays representing the ranges of each
            dimension in the input space. The length of the list should match the number of
            dimensions in the input space.
        axes (tuple of int, optional): A tuple containing the indices of the dimensions to plot on
            the x and y axes. Only applicable when the system has more than one dimension. If not
            provided, the first two dimensions (0, 1) will be plotted by default.
    """
    if len(x_ranges) == 1:
        x = torch.tensor(x_ranges[0], dtype=torch.float32)
        energy = model(x).detach().numpy()
        plt.plot(x_ranges[0], energy)
        plt.xlabel("x")
        plt.ylabel("Energy")
        plt.show()

    else:
        if axes is None:
            axes = (0, 1)

        x_meshgrid = np.meshgrid(*[x_range for i, x_range in enumerate(x_ranges) if i in axes])
        x = np.stack(x_meshgrid, axis=-1)
        x_full = np.zeros((x.shape[:-1] + (len(x_ranges),)), dtype=np.float32)
        x_full[..., axes] = x
        x_full = torch.tensor(x_full, dtype=torch.float32)

        energy = model(x_full).detach().numpy().squeeze()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(*x_meshgrid, energy, cmap=cm.viridis)
        plt.tight_layout()
        plt.savefig("energy_fxn.pdf")
        plt.close()

def plot_errors(errors, dt_values):
    colors = cm.viridis(np.linspace(0, 1, len(dt_values)))

    for dt, error, color in zip(dt_values, errors, colors):
        mean = np.mean(error, axis=1)
        std_error = np.std(error, axis=1) / np.sqrt(error.shape[1])
        plt.plot(range(1,len(mean)+1), mean, color=color, label=f'dt = {dt}')
        plt.fill_between(range(1,len(mean)+1), mean - std_error, mean + std_error, color=color, alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.savefig("errors_over_dts.pdf")
    plt.close()

def main():
    args = Args()
    model = EnergyPredictor(args)

    train(model, args, Args.dynamics_function, contrastive_loss)

    num_samples = 10000
    num_dims = 2  # Change this to the number of dimensions of your system
    x_t = torch.tensor(np.random.uniform(-5,5,(num_samples, num_dims)), dtype=torch.float32, requires_grad=True)
    dt_values = [0.01]
    num_steps = 100

    errors = []
    for dt in dt_values:
        errors.append(evaluate(model, x_t, dt, Args.dynamics_function, num_steps))

    x_ranges = [np.linspace(-5, 5, 1000) for _ in range(num_dims)]

    plot_energy(model, x_ranges)
    plot_errors(errors, dt_values)

if __name__ == "__main__":
    main()
