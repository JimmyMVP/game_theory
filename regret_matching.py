import numpy as np
from tqdm import tqdm
from collections import deque

coding = ['ROCK', 'PAPER', 'SCISSORS']
actions = [0, 1, 2]

def rps(a1, a2):
    if a1 == a2:
        return 0,0
    if (a1 - a2) in [1,2]:
        return 1, -1
    else:
        return -1, 1



p1_cumulative_regrets = np.zeros(3)
p1_strategy = np.asarray([0.2,0.5,0.3])

algo_cumulative_regrets = np.zeros(3)
algo_strategy = np.asarray([1./3] * 3)

mvavg_reward_deq = deque(maxlen=50)

def regret_matching(reward,a2,cumulative_regrets):
    """
        Returns strategy based on regret matching
    """
    missed_rewards = np.asarray([rps(a, a2)[0] for a in actions])
    regrets = missed_rewards - reward
    cumulative_regrets += regrets
    cumulative_regrets_tmp = cumulative_regrets.copy()
    cumulative_regrets_tmp[cumulative_regrets_tmp < 0] = 0

    strategy = cumulative_regrets_tmp/cumulative_regrets_tmp.sum()
    return strategy


calc_strategy_algo = regret_matching
calc_strategy_player = regret_matching
algo_cum_reward = 0

# Play 100 rounds of rock paper scissors
for i in tqdm(range(100000)):

    player_move = np.random.choice(actions, p=p1_strategy)
    algo_move = np.random.choice(actions, p=algo_strategy)

    #Calculate new strategy
    algo_reward, player_reward = rps(algo_move, player_move)
    algo_strategy = calc_strategy_algo(algo_reward, player_move, algo_cumulative_regrets)
    p1_strategy = calc_strategy_player(player_reward, algo_move, p1_cumulative_regrets)

    algo_cum_reward += algo_reward
    mvavg_reward_deq.append(algo_reward)

    if i % 10000 == 0:
        print("Algor acc reward: ", algo_cum_reward, " mvavg reward: ", np.mean(mvavg_reward_deq))

print("Algoritm cum reward: ", algo_cum_reward)