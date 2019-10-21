'''
本代码实现了智能体（红色）走迷宫到达出口（黄色）的学习过程，运用了Q-Learning算法。
没走一步的奖励为0，到达出口的奖励为1，跌入悬崖（黑色）的奖励为 -1。
'''
from maze_env import Maze
from PG import PolicyGradient


def run_maze():
	total_step=0
	for episode in range(300):
		step=0
		observation = env.reset() #初始化观测（状态）
		while True:
			env.render() #刷新环境
			action = RL.choose_action(observation)
			observation_, reward, done = env.step(action) #获取下一个状态，奖励以及是否是终点
			RL.store_transition(observation, action, reward) #存储
			observation = observation_
			if done:
				RL.learn()
				break
			step+=1
			total_step+=1
		print("\r episode: ", episode, "  step: ", step, end="")
	print("\ngame over")
	print("mean_step: ", total_step / 300)
	env.destroy()


if __name__ == "__main__":
	env = Maze()
	RL = PolicyGradient(env.n_actions,env.n_features,output_graph=True)
	env.after(100, run_maze())
	env.mainloop()
