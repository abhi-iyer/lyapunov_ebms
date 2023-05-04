from imports import *
from dynamics import *


def positive_term_loss(model, batch):
    margin = 2

    x_t = batch[:-1, :]
    x_tp1 = batch[1:, :]

    E_x_t = model(x_t)
    E_x_tp1 = model(x_tp1)

    '''
    hinge loss with margin
    '''
    loss = torch.mean(torch.maximum(torch.zeros(x_t.shape[0]).cuda(), margin + (E_x_tp1 - E_x_t))**2)

    '''
    exponential loss
    '''
    # loss = torch.mean(torch.exp(E_x_tp1 - E_x_t))

    return loss


def negative_term_loss(model, batch, num_negative_samples):
    margin = 2
    beta = 2

    losses = []


    '''
    negative samples: previous `num_negative_samples` in the trajectory given current x_t (i.e. x_{t - num_negative_samples} ... x_{t-1})
    '''
    for i in range(1, batch.shape[0]-1):
        num_samples = i - max(0, i - num_negative_samples)

        E_x_t = model(batch[i, :].unsqueeze(0)).repeat(num_samples)
        E_x_tp1 = model(batch[i - num_samples : i, :])
        
        '''
        hinge loss with margin
        '''
        loss = torch.sum(torch.maximum(torch.zeros(num_samples).cuda(), margin - (E_x_tp1 - E_x_t))**2)

        '''
        Gibbs partition function based loss
        '''
        # loss = 1/beta * torch.log(torch.sum(torch.exp(-beta * (E_x_tp1 - E_x_t))))

        '''
        negative exponential loss
        '''
        # loss = torch.sum(torch.exp(-(E_x_tp1 - E_x_t)))

        losses.append(loss)


    '''
    different way to generate negative samples:
        create a hypersphere of radius || E_x_tp1 - E_x_t ||
        pick vectors in that hypersphere that are `angle` away from vector E_x_tp1 - E_x_t
    '''
    # for i in range(batch.shape[0]-1):
    #     E_x_t = model(batch[i, :].unsqueeze(0)).repeat(num_negative_samples)
    #     E_x_tp1 = model(generate_negative_samples(batch[i, :], angle=30, num_samples=num_negative_samples))
        
    #     loss = torch.sum(torch.maximum(torch.zeros(num_samples).cuda(), margin - (E_x_tp1 - E_x_t))**2)
    #     # loss = 1/beta * torch.log(torch.sum(torch.exp(-beta * (E_x_tp1 - E_x_t))))
    #     # loss = torch.sum(torch.exp(-(E_x_tp1 - E_x_t)))

    #     losses.append(loss)

    return sum(losses)/len(losses)


def epoch(iterations, model, optimizer, data, bs, num_negative_samples):
    losses = []

    for i in range(iterations):
        batch = generate_batch(data, bs=bs)

        optimizer.zero_grad()

        loss = positive_term_loss(model, batch) + negative_term_loss(model, batch, num_negative_samples=num_negative_samples)

        loss.backward()

        optimizer.step()

        losses.append(loss.detach().cpu().item())

    return np.mean(losses)


def plot_energy_along_trajectory(model, trajectory):
    with torch.no_grad():
        E_x_t = []

        for x_t in trajectory:
            E_x_t.append(model(x_t.unsqueeze(0)).squeeze().detach().cpu().item())

    plt.plot(E_x_t)

    plt.show()


def plot_energy_landscape(model):
    X = torch.linspace(-10, 10, steps=500)
    Y = torch.linspace(-10, 10, steps=500)

    energy_manifold = model(torch.cartesian_prod(X, Y).cuda()).reshape(X.shape[0], Y.shape[0]).detach().cpu().numpy()

    X, Y = np.meshgrid(X.numpy(), Y.numpy())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(X, Y, energy_manifold, cmap='viridis', edgecolor='k', linewidth=0.5, facecolor=(1,1,1,0.2))

    fig.colorbar(surface, pad=0.1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$energy$')
    ax.set_title('energy landscape')
    
    plt.show()