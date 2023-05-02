import numpy as np
import torch 

def linear_dynamics(x_t, dt):
    """A linear dynamics function for an n-dimensional system.

    Args:
        x_t (torch.Tensor): The current state tensor.
        dt (float): The time step.

    Returns:
        torch.Tensor: The next state tensor after applying linear dynamics.
    """
    linear_transformation_matrix = np.array([
           [-1, 0], 
            [0, -2]
        ]) 
    if isinstance(x_t,np.ndarray):
        x_t = torch.tensor(x_t,dtype=torch.float32)
    matrix = torch.tensor(linear_transformation_matrix, dtype=torch.float32)
    x_tp1 = x_t + dt * x_t @ matrix
    return x_tp1

def linear_1d(x_t,dt):
    return x_t + dt * -x_t