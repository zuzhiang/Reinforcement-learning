import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN():
    def __init__(self,
                 n_actions,
                 n_states,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=2000,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 double_q=True
                 ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q  # 决定是否使用 double q

        self.learn_step_counter = 0  # 学习花费的步数
        # 记忆库的大小，用来保存 [s,a,r,s_]，一个状态由两个数字来表示
        self.memory = np.zeros((self.memory_size, n_states * 2 + 2))
        self.build_net()  # 创建两个神经网络

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")  # 目标网络的参数
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")  # 估值网络的参数

        with tf.variable_scope("hard_replacement"):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # tf.assign(a,b) 可以将 b 的值赋给 a
        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 历史损失

    def build_net(self):
        # 以下是神经网络所有的输入
        self.s = tf.placeholder(tf.float32, [None, self.n_states], name="s")
        self.s_ = tf.placeholder(tf.float32, [None, self.n_states], name="s_")
        self.r = tf.placeholder(tf.float32, [None, ], name="r")
        self.a = tf.placeholder(tf.int32, [None, ], name="a")
        # 初始化权重和偏置
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope("eval_net"):  # 估值网络
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="e1")  # 第1层
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name="e2")  # 第二层
            # Q估计中每个状态的动作值最大的下标
            action_index = tf.argmax(self.q_eval,axis=1,output_type =  tf.int32)

        with tf.variable_scope("target_net"):  # 目标网络
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="t1")
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name="t2")

        with tf.variable_scope("q_target"):
            # 下面的语句不可换为：self.q_next.shape[0]，因为该语句是静态获取shape
            row=tf.range(tf.shape(self.q_next)[0])
            pos = tf.stack([row,action_index],axis=1)
            '''
            tf.gather(data,indices,axis): 在data张量的axis对应的轴中，按照下标数组
            indices选取元素。
            tf.gather_nd(data,indices): 在data张量中，按照下标数组indices选取元素，
            其中indices是data的前几个维度。
           '''
            val = tf.gather_nd(params=self.q_next,indices=pos)
            q_target = self.r + self.gamma * val # Q 现实
            self.q_target = tf.stop_gradient(q_target)  # 对 q_target的反向传播进行截断
        with tf.variable_scope("q_eval"):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)  # 动作的下标
            # 根据索引提取 params 中的元素，构建新的 tensor
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # Q估计
        with tf.variable_scope("loss"):  # 损失函数，本质是 Q 表的更新公式
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name="TD_error"))
        with tf.variable_scope("train"):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):  # 存储
        if not hasattr(self, "memory_counter"):  # hasattr() 函数用于判断对象是否包含对应的属性
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):  # 选择行动
        observation = observation[np.newaxis, :]  # 增加一个新的维度
        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)  # 随机选择行动
        else:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)  # 选择值最大的行为
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:  # 隔一段时间更新目标网络
            self.sess.run(self.target_replace_op)
        # print("\ntarget_params_replaced\n")
        if self.memory_counter > self.memory_size:  # 如果记忆库超出最大容量
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 对神经网络进行训练并得到损失
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_states],
                self.a: batch_memory[:, self.n_states],
                self.r: batch_memory[:, self.n_states + 1],
                self.s_: batch_memory[:, -self.n_states:]
            })
        self.cost_his.append(cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # 设置 epsilon 是递增的（不超过最大值）
        self.learn_step_counter += 1

    def plot_cost(self):  # 绘制损失图
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel("Cost")
        plt.xlabel("training steps")
        plt.show()
