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


def dampened_oscillator(m, c, k, seed):
    def func(t, state, m, c, k):
        dim = len(c)
        state = state.reshape(dim, 2)

        x, x_dot = state[:, 0], state[:, 1]

        # c = dampening coefficients
        # k = stiffness coefficients
        x_dot_dot = (-c*x_dot - k*x) / m

        return np.column_stack((x_dot, x_dot_dot)).flatten()
    
    dim = len(c)
    
    # initial conditions
    torch.manual_seed(seed)
    x0 = torch.rand(dim * 2).numpy()

    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    solution = solve_ivp(func, t_span, x0, args=(m, c, k), t_eval=t_eval, rtol=1e-6, atol=1e-6)

    xs = []
    x_dots = []
    for i in range(dim):
        xs.append(solution.y[2*i, :])
        x_dots.append(solution.y[2*i + 1, :])

    return torch.Tensor(np.hstack((np.column_stack(xs), np.column_stack(x_dots)))).cuda(), t_span[1]/len(t_eval)


def vdp_oscillator(dim, mu, seed):
    def van_der_pol_oscillator(dim, mu):
        def vdp_equations(y, t, mu):
            dydt = np.zeros(dim * 2)
            for i in range(0, dim * 2, 2):
                dydt[i] = y[i + 1]
                dydt[i + 1] = mu * (1 - y[i]**2) * y[i + 1] - y[i]
            return dydt

        return vdp_equations

    # initial conditions
    torch.manual_seed(seed)
    x0 = torch.rand(dim * 2).numpy()

    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # solve the Van der Pol oscillator differential equations
    vdp = van_der_pol_oscillator(dim, mu)
    solution = odeint(vdp, x0, t_eval, args=(mu,))

    xs = []
    x_dots = []
    for i in range(dim):
        xs.append(solution[:, 2*i])
        x_dots.append(solution[:, 2*i + 1])

    return torch.Tensor(np.hstack((np.column_stack(xs), np.column_stack(x_dots)))).cuda(), t_span[1]/len(t_eval)


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
    plt.tight_layout()
    ax = plt.axes(projection='3d')
 
    ax.plot3D(np.arange(x.shape[0]), x[:, 0], x[:, 1], alpha=1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("$x$")
    ax.set_zlabel("$\dot{x}$")

 
def plot_3d_trajectory(actual, pred):
    fig = plt.figure()
    plt.tight_layout()
    ax = plt.axes(projection='3d')

    ax.plot3D(np.arange(actual.shape[0]), actual[:, 0], actual[:, 1], alpha=1.0, color='blue', label='true')
    ax.plot3D(np.arange(pred.shape[0]), pred[:, 0], pred[:, 1], alpha=1.0, color='purple', label='pred')

    ax.set_xlabel("time")
    ax.set_ylabel("$x$")
    ax.set_zlabel("$\dot{x}$")

    ax.legend()