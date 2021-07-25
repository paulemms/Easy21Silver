import numpy as np
import environment as env


def monte_carlo(num_episodes=20000, n0=100, exact_q=None):

    e = env.Easy21Environment()
    model = env.Model()

    # hold the policy in tabular form
    pi = np.full(e.state_size, env.Action.HIT)  # e-greedy means starting policy irrelevant
    n = np.zeros(e.state_action_size, dtype=int)
    ns = np.zeros(e.state_size, dtype=int)
    q = np.zeros(e.state_action_size)
    model.q_old = q.copy()

    # monte-carlo
    for i in range(num_episodes):

        # simulate episode
        ep = e.get_episode(pi)
        g = np.cumsum([r for _, _, r in ep][::-1])

        # GLIE - greedy in the limit with infinite exploration
        for t, ((sd, sp), a, r) in enumerate(ep):

            n[sd, sp, a.value] += 1
            ns[sd, sp] += 1

            alpha = 1 / n[sd, sp, a.value]
            q[sd, sp, a.value] += alpha * (g[t] - q[sd, sp, a.value])

        # improve policy using vectorised e-greedy
        pi[1:, 1:] = np.array([*env.Action], object)[np.argmax(q[1:, 1:], axis=2)]  # greedy
        epsilon = n0 / (n0 + ns)
        is_random = np.random.rand(*epsilon.shape) <= epsilon  # choose random action with prob eps, ignore zero indices
        pi[is_random] = np.array([*env.Action], object)[np.random.randint(0, high=2, size=np.count_nonzero(is_random))]

        # stats
        if r == 1:
            model.num_wins += 1
        if i > 0 and (i+1) % 10000 == 0:
            model.append_diagnostics(i + 1, q, exact_q)

    model.final_q = q
    return model


