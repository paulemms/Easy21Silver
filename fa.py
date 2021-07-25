import numpy as np
import environment as env
from numpy.random import randint
from numpy.random import uniform

# feature mapping
feature_size = [3, 6, 2]
dealer = [[1, 4], [4, 7], [7, 10]]
player = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]

e = env.Easy21Environment()
dealer_range = range(1, e.max_initial_dealer_card + 1)
player_range = range(1, e.max_player_sum + 1)
dealer_pos = [[j for j, (a, b) in enumerate(dealer) if a <= s0 <= b] for s0 in dealer_range]
player_pos = [[j for j, (a, b) in enumerate(player) if a <= s0 <= b] for s0 in player_range]
action_pos = [[j for j, a in enumerate(env.Action) if a == act] for act in env.Action]


def e_greedy_lfa(theta, s, eps):
    """Choose action a from s using linear value function approximation"""
    if uniform() <= eps:
        a = env.Action(randint(len(env.Action)))
    else:
        q = [np.dot(get_phi(s, a), theta) for a in env.Action]
        a = env.Action(np.argmax(q))  # greedy

    return a


def get_phi(s, a):
    vec = np.zeros(feature_size, dtype=int)

    for d in dealer_pos[s[0]-1]:
        for p in player_pos[s[1]-1]:
            for aa in action_pos[a.value]:
                vec[d][p][aa] = 1

    return vec.flatten()


def expand_q(theta):
    """Remaps the linear feature representation of q into (s, a) space"""
    q = np.zeros(e.state_action_size)

    for d in range(1, e.max_initial_dealer_card + 1):
        for p in range(1, e.max_player_sum + 1):
            for a in env.Action:
                s = (d, p)
                q[d, p, a.value] = np.dot(get_phi(s, a), theta)

    return q


def lfa(num_episodes=20000, la=0, eps=0.05, alpha=0.01, exact_q=None):
    """Linear function approximation"""

    model = env.Model()

    num_features = np.prod(feature_size)
    theta = np.zeros(num_features)
    eligibility = np.zeros(num_features)  # eligibility trace
    model.q_old = expand_q(theta).copy()

    for i in range(num_episodes):

        # initialise eligibility trace
        eligibility[:] = 0

        # start episode
        s = e.get_initial_state()
        a = e_greedy_lfa(theta, s, eps)

        # repeat until episode terminates
        while s is not None:

            # get parameters that represent this state and action
            phi = get_phi(s, a)
            q = np.dot(phi, theta)

            # update counters corresponding to s, a in feature space
            eligibility += phi

            # take action a and observe r, sd
            sd, r = e.step(s, a)

            # TD error
            if sd is None:  # episode terminates
                delta = r - q
            else:
                ad = e_greedy_lfa(theta, sd, eps)
                phi = get_phi(sd, ad)
                qd = np.dot(phi, theta)
                delta = r + qd - q
                a = ad

            # update theta in proportion to td error delta and eligibility trace e
            theta += alpha * delta * eligibility
            eligibility *= la

            # update state
            s = sd

        # stats
        if r == 1:
            model.num_wins += 1
        if i > 0 and (i+1) % 10000 == 0:
            model.append_diagnostics(i + 1, expand_q(theta), exact_q)

    model.final_q = expand_q(theta)
    return model




