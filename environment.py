from enum import Enum
import numpy as np
import pandas as pd
from numpy.random import randint
from numpy.random import uniform


class Action(Enum):
    HIT = 0
    STICK = 1


class Colour(Enum):
    RED = 0
    BLACK = 1


class Easy21Environment:
    max_initial_dealer_card = 10
    max_player_sum = 21
    state_size = [max_initial_dealer_card + 1, max_player_sum + 1]  # ignore 0 index
    state_action_size = state_size + [len(Action)]

    def __init__(self):
        self.dealer_first_card = None
        self.player_first_card = None

    def get_random_initial_state(self):
        """Get random initial state"""
        self.dealer_first_card, _ = self.get_new_card()
        self.player_first_card, _ = self.get_new_card()

        return self.dealer_first_card, self.player_first_card

    def step(self, s, a):
        """Returns the next state given state s and action a or None if state terminates"""
        dealer_first_card, player_sum = s
        r = 0

        if a is Action.HIT:
            player_num, player_col = self.get_new_card()
            player_sum += player_num if player_col is Colour.BLACK else -player_num
            if 0 < player_sum <= 21:
                next_state = (dealer_first_card, player_sum)
            else:
                r = -1
                next_state = None
        elif a is Action.STICK:
            # dealer starts taking turns
            dealer_sum = dealer_first_card
            while 0 < dealer_sum < 17:
                dealer_num, dealer_col = self.get_new_card()
                dealer_sum += dealer_num if dealer_col is Colour.BLACK else -dealer_num
            if dealer_sum > 21 or dealer_sum < 1 or player_sum > dealer_sum:
                r = 1
            elif dealer_sum > player_sum:
                r = -1
            next_state = None

        else:
            raise Exception('Unknown action {}'.format(a))

        return next_state, r

    def get_episode(self, policy, initial_state=None):
        """Get episode from a given or random initial state"""
        episode = list()
        if initial_state is None:
            s = self.get_random_initial_state()
        else:
            s = initial_state
        while s is not None:
            a = policy[s]
            s_dash, reward = self.step(s, a)
            episode.append((s, a, reward))
            s = s_dash
        return episode

    def reward_dist(self, pi, reward=1, num_samples=10000):
        """Expected reward given a policy and the initial state"""

        mean_reward = np.zeros((self.max_initial_dealer_card + 1,
                                self.max_initial_dealer_card + 1))
        for d in range(1, self.max_initial_dealer_card + 1):
            for p in range(1, self.max_initial_dealer_card + 1):
                s = (d, p)
                num_rewards = 0

                for i in range(num_samples):
                    ep = self.get_episode(pi, initial_state=s)
                    r = ep[-1][2]
                    if r == reward:
                        num_rewards += 1

                mean_reward[s] = num_rewards / num_samples

        return mean_reward

    @staticmethod
    def get_new_card():
        """Get new card from deck of cards"""
        num = randint(1, 11)  # between 1 and 10 inc, slightly slower than random.random for scalars
        col = Colour.RED if uniform() < 1 / 3.0 else Colour.BLACK

        return num, col


class Model:
    """
    Model of an environment.
    Contains a dataFrame of diagnostics gathered during training.
    """

    def __init__(self, show_diagnostics=False):
        self.show_diagnostics = show_diagnostics

        self.num_wins = 0
        self.q_old = None
        self.final_q = None

        self.df = pd.DataFrame({
            'Episode': pd.Series([], dtype='int'),
            'Relative_Error': pd.Series([], dtype='float'),
            'Percent_Wins': pd.Series([], dtype='float'),
            'MSE': pd.Series([], dtype='float')
        })

    def append_diagnostics(self, i, q, exact_q=None):
        percent_wins = float(self.num_wins) / (i + 1) * 100.0
        rel_error = np.linalg.norm(q - self.q_old)
        self.q_old = q.copy()
        row = {'Episode': i, 'Relative_Error': rel_error, 'Percent_Wins': percent_wins}
        if exact_q is not None:
            row['MSE'] = np.square(q - exact_q).mean()
        if self.show_diagnostics:
            print(row)
        self.df = self.df.append(row, ignore_index=True)


def test_policy(first_dealer_card, player_sum):
    if player_sum > 16:
        return Action.STICK
    else:
        return Action.HIT


def random_policy(first_dealer_card, player_sum):
    if np.random.uniform() < 0.5:
        return Action.HIT
    else:
        return Action.STICK


def e_greedy(q, s, n0, n):
    """Choose action a from s using policy derived from q"""
    ns = np.sum(n[s]) + 1  # have visited s now
    eps = n0 / (n0 + ns)
    if uniform() <= eps:
        a = Action(randint(len(Action)))
    else:
        a = Action(np.argmax(q[s]))  # greedy

    return a

