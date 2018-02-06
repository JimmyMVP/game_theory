from blist import sorteddict
from collections import namedtuple
import numpy as np
from tqdm import tqdm
import random

"""
 Implementation of counterfactual regret minimization
"""

PASS = 0
BET = 1
NUM_ACTIONS = 2

Node = namedtuple("Node", ["regret_sum", "strategy", "strategy_sum"])

information_sets = sorteddict()


class Node(object):

    def __init__(self, info_set):

        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)
        self.info_set = info_set

    def get_strategy(self, realization_weight):
        strat = self.regret_sum.copy()
        strat[strat>0] = strat[strat>0].sum()
        strat[strat<=0] = 1./NUM_ACTIONS
        self.strategy_sum += realization_weight * strat

        return strat

    def get_average_strategy(self):
        nomalising_sum = self.strategy_sum.sum()
        return self.strategy_sum/nomalising_sum if nomalising_sum > 0 else np.repeat(1./NUM_ACTIONS, NUM_ACTIONS)

    def __str__(self):
        return "{:4s} {}".format(hash(self), str(self.get_average_strategy()))


def cfr(cards, history, p0, p1):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player
    # Return payoff for terminal state
    if plays > 1:
        terminal = history[plays-1] is "p"
        double_bet = history[plays-2::plays] is "bb"
        is_player_card_higher = cards[player] > cards[opponent]

        if terminal:
            if history is "pp":
                return 1 if is_player_card_higher else -1
            else:
                return 1
        elif double_bet:
            return 2 if is_player_card_higher else -2

    information_set = str(cards[player]) + history
    if information_set in information_sets:
        node = information_sets[information_set]
    else:
        node = Node(information_set)

    strategy = node.get_strategy(p0 if player == 0 else p1)
    util = np.zeros(NUM_ACTIONS)
    node_util = 0

    for a in range(NUM_ACTIONS):
        next_history = history + "p" if a == 0 else "b"
        util[a] = -cfr(cards, next_history, p0*strategy[a], p1) if player == 0 \
            else -cfr(cards, next_history, p0, p1*strategy[a])
        node_util += strategy[a] * util[a]

    for a in range(NUM_ACTIONS):
        regret = util[a] - node_util
        node.regret_sum[a] += p1*regret if player == 0 else p0*regret

    return node_util


def train(iterations):

    cards = [1,2,3]
    util = 0.
    for i in tqdm(range(iterations)):
        #Shuffle cards
        np.random.shuffle(cards)
        util += cfr(cards, "", 1, 1)

    print("Average game value: " + util/i)
    for n in information_sets.values():
        print(n)



train(10000)