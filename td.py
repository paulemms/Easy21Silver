import numpy as np
import environment as env


def sarsa0(num_episodes=20000, n0=100, exact_q=None):
    """Simple SARSA with a single Bellman backup"""

    e = env.Easy21Environment()
    model = env.Model()

    n = np.zeros(e.state_action_size, dtype=int)
    q = np.zeros(e.state_action_size)
    model.q_old = q.copy()

    for i in range(num_episodes):

        # start episode
        s = e.get_initial_state()
        a = env.e_greedy(q, s, n0, n)

        # repeat until episode terminates
        while s is not None:

            # update counters
            n[s[0], s[1], a.value] += 1
            alpha = 1 / n[s[0], s[1], a.value]

            # take action a and observe r, sd
            sd, r = e.step(s, a)

            # update q using Bellman backup
            if sd is None:  # episode terminates
                q[s[0], s[1], a.value] += alpha * (r - q[s[0], s[1], a.value])
            else:
                ad = env.e_greedy(q, sd, n0, n)
                target = r + q[sd[0], sd[1], ad.value]
                q[s[0], s[1], a.value] += alpha * (target - q[s[0], s[1], a.value])
                a = ad

            # update state
            s = sd

        # stats
        if r == 1:
            model.num_wins += 1
        if i > 0 and (i+1) % 10000 == 0:
            model.append_diagnostics(i + 1, q, exact_q)

    model.final_q = q
    return model


def sarsa(num_episodes=20000, n0=100, la=1, exact_q=None):
    """SARSA(lambda)"""

    e = env.Easy21Environment()
    model = env.Model()

    n = np.zeros(e.state_action_size, dtype=int)
    q = np.zeros(e.state_action_size)
    model.q_old = q.copy()
    eligibility = np.zeros(e.state_action_size)

    for i in range(num_episodes):

        # initialise eligibility trace
        eligibility[:] = 0

        # start episode
        s = e.get_initial_state()
        a = env.e_greedy(q, s, n0, n)

        # repeat until episode terminates
        while s is not None:

            # update counters
            n[s[0], s[1], a.value] += 1
            eligibility[s[0], s[1], a.value] += 1
            alpha = 1 / n[s[0], s[1], a.value]

            # take action a and observe r, sd
            sd, r = e.step(s, a)

            # TD error
            if sd is None:  # episode terminates
                delta = r - q[s[0], s[1], a.value]
            else:
                ad = env.e_greedy(q, sd, n0, n)
                delta = r + q[sd[0], sd[1], ad.value] - q[s[0], s[1], a.value]
                a = ad

            # update q in proportion to td error delta and eligibility trace e
            q += alpha * delta * eligibility
            eligibility *= la

            # update state
            s = sd

        # stats
        if r == 1:
            model.num_wins += 1
        if i > 0 and (i+1) % 10000 == 0:
            model.append_diagnostics(i + 1, q, exact_q)

    model.final_q = q
    return model
