import numpy as np
import pandas as pd

class QLearningTable():
	def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
		self.actions=actions
		self.lr=learning_rate
		self.gamma=reward_decay
		self.epsilon=e_greedy
		self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64)

	def chooce_action(self,observation): #选择行动
		print("observation=", observation)
		self.check_state_exist(observation)
		if np.random.uniform()>self.epsilon:
			action=np.random.choice(self.actions) #随机选择行动
		else:
			state_list=self.q_table.loc[observation,:]
			action=np.random.choice(state_list[state_list==np.max(state_list)].index)
			# 选择Q表值最大的行动，如果存在多个最大值，则在里面随机选一个
		return action

	def update_q_table(self,s,a,r,s_): # 更新 Q 表
		self.check_state_exist(s_)
		q_predict = self.q_table.loc[s, a]
		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.loc[s_, :].max()
		else:  # 终止状态
			q_target = r
		self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 根据公式更新

	def check_state_exist(self,state): #检测状态是否存在
		if state not in self.q_table.index: #如果不存在则增加该状态
			self.q_table = self.q_table.append(
				pd.Series(
					[0] * len(self.actions),
					index=self.q_table.columns,
					name=state,
				)
			)