'''
本代码实现了智能体（红色）走迷宫到达出口（黄色）的学习过程，运用了Sarsa算法。
每走一步的奖励为0，到达出口的奖励为1，跌入悬崖（黑色）的奖励为 -1。
'''
from maze_env import Maze
from sarsa import SarsaTable


def Sarsa():
	total_step=0
	for episode in range(25):
		step=0
		observation = env.reset() #初始化观测（状态）
		action=RL.choose_action(str(observation)) #初始化动作
		while True:
			env.render() #刷新环境
			observation_, reward, done = env.step(action) #获取下一个状态，奖励以及是否是终点
			action_ = RL.choose_action(str(observation_)) #选择下一个动作
			RL.update_q_table(str(observation), action, reward, str(observation_),action_)
			action=action_
			observation = observation_
			if done:
				break
			step += 1
			total_step += 1
		print("\r episode: ", episode, "  step: ", step, end="")
	print("\ngame over")
	print("mean_step: ", total_step / 25)
	env.destroy()

if __name__ == "__main__":
	env = Maze()
	RL = SarsaTable(actions=list(range(env.n_actions)))
	env.after(100, Sarsa)
	env.mainloop()
