import timeit
import numpy as np
import mc
import td
import fa
import plots as plt


def timings(num_episodes=10000, repetitions=10):
    mc_results = timeit.timeit(lambda: mc.monte_carlo(num_episodes=num_episodes), number=repetitions, globals=globals())
    print('Monte-carlo: {:6.2f}'.format(mc_results))
    sarsa0_results = timeit.timeit(lambda: td.sarsa0(num_episodes=num_episodes), number=repetitions, globals=globals())
    print('SARSA0:\t\t {:6.2f}'.format(sarsa0_results))
    sarsa_results = timeit.timeit(lambda: td.sarsa(num_episodes=num_episodes), number=repetitions, globals=globals())
    print('SARSA:\t\t {:6.2f}'.format(sarsa_results))


if __name__ == "__main__":

    # fix seed - there's a newer way to do this that handles concurrency
    np.random.seed(100)

    # timings()
    plt.standard_plots()

    # plt.plot_sarsa_lambda_mse(alg=td.sarsa, title='MSE of SARSA(lambda)')
    # plt.plot_mse_episode(alg=td.sarsa, title='MSE of SARSA(lambda)', lambdas=[0.0, 1.0])

    # plt.plot_sarsa_lambda_mse(alg=fa.lfa, title='MSE of SARSA(lambda) with LFA')
    # plt.plot_mse_episode(alg=fa.lfa, title='MSE of SARSA(lambda) with LFA', lambdas=[0.0, 1.0])


