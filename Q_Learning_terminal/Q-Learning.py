'''
本代码实现了一个智能体（用字符 'o' 表示）寻找宝藏（用字符 'T' 表示）
的强化学习过程，所用的算法是 Q-Learning 算法。当智能体找到宝藏时奖励
值为 1，反之为 0。智能体的动作只有两个—— left 和 right。程序会显示
智能体和宝藏的位置，每个回合的奖励值，以及终止状态时 Q 表的情况。
'''
import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 10  # 回合数
FRESH_TIME = 0.3


def build_q_table(n_states, actions):  # 初始化 Q 表
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,  # 列的名称
    )
    return table


def choose_action(state, q_table):  # 根据当前状态和 Q 表选择一个动作
    # 如果随机数的值大于 EPSIlON 或者 Q 表的当前列都是 0 （即第一次到达该状态）
    if (np.random.uniform() > EPSILON or (q_table.iloc[state, :] == 0).all()):
        action_name = np.random.choice(ACTIONS)  # 随机选择一个动作
    else:
        action_name = q_table.iloc[state, :].idxmax()  # 选择当前状态对应的值最大的动作
    return action_name


def get_env_feedback(S, A):  # 获取新的状态和对应的奖励
    R = 0
    if A == "right":  # 往右
        S_ = S + 1
    else:  # 往左
        S_ = S - 1
        if S_ < 0:  # 如果到达最左端
            S_ = 0
    if S_ == N_STATES - 1:  # 如果找到宝藏
        S_ = "terminal"
        R = 1
    return S_, R


def update_env(S, episode, step_counter):  # 更新环境，主要是用来显示，可以不必搞懂
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == "terminal":
        interaction = "Episode %s: total_steps = %s" % (episode + 1, step_counter)
        print("\r{}".format(interaction), end="")
        time.sleep(2)
        print("\r                                  ", end="")
    else:
        env_list[S] = 'o'
        interaction = "".join(env_list)
        print("\r{}".format(interaction), end="")
        time.sleep(FRESH_TIME)


def q_learning():  # Q-Learning 算法
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        S = 0
        step_counter = 0 # 共走了多少步
        is_terminal = False
        update_env(S, episode, step_counter)
        while not is_terminal:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            # 下面的代码是为了更新 Q 表
            q_predict = q_table.loc[S, A] # Q 估计
            if S_ != 'terminal':
				# Q 现实
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:  # 终止状态
                q_target = R
                is_terminal = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # 根据公式更新
            S = S_
            step_counter += 1
            update_env(S, episode, step_counter)
    return q_table


if __name__ == "__main__":
    q_table = q_learning()
    print("\r\nQ-table:\n")
    print(q_table)
