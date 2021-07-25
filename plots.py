import pdb

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm
import mc
import td
import fa


def standard_plots(num_episodes=100000):
    """Plots of the value function and optimal policy for each algorithm"""

    # algorithms
    mc_model = plot_value_policy(mc.monte_carlo(num_episodes))
    sarsa0_model = plot_value_policy(td.sarsa0(num_episodes))
    sarsa_model = plot_value_policy(td.sarsa(num_episodes, la=0.5))
    lfa_model = plot_value_policy(fa.lfa(num_episodes, la=0.0))


def plot_sarsa_lambda_mse(alg, title, num_episodes=10000):
    """
    Plot the MSE using SARSA with lambda = {0, 0.1, 0,2, ..., 1}

    :param int num_episodes: Number of episodes for each SARSA run
    """
    pyplot.figure()
    lambdas = np.arange(0, 11) / 10

    print('Calculating exact solution using monte-carlo')
    exact_model = mc.monte_carlo(num_episodes=100000)
    exact_q = exact_model.final_q

    mse = list()
    for la in lambdas:
        np.random.seed(100)  # fix same random sequence for each model
        print(f'Training SARSA({la})')
        model = alg(num_episodes=num_episodes, la=la)
        err = np.square(model.final_q - exact_q).mean()
        mse.append(err)

    pyplot.plot(lambdas, mse, 'o-')
    pyplot.xlabel("lambda")
    pyplot.ylabel("MSE")
    pyplot.title(title + f" after {num_episodes:d} episodes")

    pyplot.show()


def plot_mse_episode(alg, title, lambdas, num_episodes=10000):
    """
    Plot the MSE using SARSA for each given lambda

    :param list[float] lambdas: list of lambda values
    :param int num_episodes: Number of episodes for each SARSA run
    """
    pyplot.figure()

    print('Calculating exact solution using monte-carlo ...')
    exact_model = mc.monte_carlo(num_episodes=100000)
    exact_q = exact_model.final_q

    for la in lambdas:
        print(f'Training SARSA({la})')
        np.random.seed(100)  # fix same random sequence for each model
        model = alg(num_episodes=num_episodes, la=la, exact_q=exact_q)
        pyplot.plot(model.df['Episode'], model.df['MSE'], 'o-', label='Lambda=' + str(la))

    pyplot.xlabel("Episode number")
    pyplot.ylabel("MSE")
    pyplot.legend()
    pyplot.title(title + " per episode")

    pyplot.show()


def plot_value_policy(model):
    """
    Plot value function and policy given the q-function

    :param Model model: Trained model object
    """
    v = np.max(model.final_q[1:, 1:], axis=2)
    pi = np.argmax(model.final_q[1:, 1:], axis=2)

    # plot policy
    fig_pi = pyplot.figure()
    # fig_pi.canvas.manager.window.move(100, 100)
    fig_pi.add_subplot(111)
    centers = [1, pi.shape[1], 1, pi.shape[0]]
    dx, = np.diff(centers[:2]) / (pi.shape[1] - 1)
    dy, = -np.diff(centers[2:]) / (pi.shape[0] - 1)
    extent = [centers[0] - dx / 2, centers[1] + dx / 2, centers[2] + dy / 2, centers[3] - dy / 2]
    pyplot.imshow(pi, cmap='tab10', interpolation='nearest', extent=extent)
    pyplot.xticks(np.arange(1, pi.shape[1] + 1, dtype=np.int))

    # plot value function
    fig = pyplot.figure()
    # fig.canvas.manager.window.move(200, 200)
    ax = fig.add_subplot(111, projection='3d')

    # Make grid offset by one to reflect card values
    x = np.arange(1, v.shape[1] + 1)
    y = np.arange(1, v.shape[0] + 1)
    x, y = np.meshgrid(x, y)

    surf = ax.plot_surface(x, y, v, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    pyplot.show()

    return model
