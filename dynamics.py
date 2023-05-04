from imports import *


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


def dampened_oscillator(m, c, k):
    def func(t, state, m, c, k):
        dim = len(c)
        state = state.reshape(dim, 2)

        x, x_dot = state[:, 0], state[:, 1]

        # c = dampening coefficients
        # k = stiffness coefficients
        x_dot_dot = (-c*x_dot - k*x) / m

        return np.column_stack((x_dot, x_dot_dot)).flatten()
    
    dim = len(c)
    
    x0 = np.array([1.0,] * dim)
    x_dot0 = np.array([0.0,] * dim)
    state = np.column_stack((x0, x_dot0)).flatten()

    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    solution = solve_ivp(func, t_span, state, args=(m, c, k), t_eval=t_eval, rtol=1e-6, atol=1e-6)

    xs = []
    for i in range(dim):
        xs.append(solution.y[2*i])

    return torch.Tensor(np.column_stack(xs)).cuda()


def vdp_oscillator():
    pass


def generate_batch(x, bs):
    i = np.random.randint(0, x.shape[0] - bs + 1, size=(1,)).item()

    return x[i : i + bs, :]


def generate_negative_samples(x, angle, num_samples):
    magnitude = torch.linalg.norm(x)

    samples = []

    while len(samples) < num_samples:
        rand_vector = torch.rand(x.numel()).cuda()
        rand_vector /= torch.linalg.norm(rand_vector)

        if torch.dot(x/magnitude, rand_vector) < np.cos(np.deg2rad(angle)):
            samples.append(rand_vector * magnitude)

    return torch.stack(samples).cuda()


def plot_3d(x):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
 
    ax.plot3D(np.arange(x.shape[0]), x[:, 0], x[:, 1], alpha=1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("dim 1")
    ax.set_zlabel("dim 2")