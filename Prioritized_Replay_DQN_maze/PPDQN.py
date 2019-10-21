import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0  # self.data 数组的指针

    def __init__(self, capacity):
        self.capacity = capacity  # 总样本数
        '''
        因为 SumTree 是一棵完全二叉树，所以可以用数组存储，如果一个节点
        的下标为 i，则其左右儿子节点的下标分别为 2*i 和 2*i+1。SumTree 
        中叶节点存储样本优先级，内部节点存储其左右儿子节点优先级之和。
        '''
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # 存储样本

    def add(self, p, data):  # 更新样本及SumTree的优先级，p是样本的优先级
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # 更新样本值
        self.update(tree_idx, p)  # 更新SumTree中的优先级

        # 以下代码实现了只保存最新的若干样本
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):  # 更新SumTree的优先级
        change = p - self.tree[tree_idx]  # 样本优先级改变之后的误差
        self.tree[tree_idx] = p
        while tree_idx != 0:  # 对当前节点所有的祖先节点更新误差
            tree_idx = (tree_idx - 1) // 2  # // 是向下取整
            self.tree[tree_idx] += change

    def get_leaf(self, v):  # 获取叶子节点的下标，其对应的样本值和样本优先级
        parent_idx = 0  # 从根节点开始
        while True:
            left = 2 * parent_idx + 1  # 当前节点的左儿子和右儿子
            right = left + 1
            if left >= len(self.tree):  # 搜索结束
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left]:  # 往左子树搜索
                    parent_idx = left
                else:  # 往右子树搜索，并更新搜索值
                    v -= self.tree[left]
                    parent_idx = right

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):  # 获取总优先级
        return self.tree[0]  # 根节点的值


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # 对误差绝对值进行截断

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):  # 存储样本并更新SumTree，transition的格式是( s, a, r, s_ )
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  # 获得历史样本的最大优先级
        if max_p == 0:  # 如果最大优先级为0，即无历史样本时
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):  # 采n个样本
        # n个样本的下标，n个样本值，
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # 将总优先级分为n个区间，在每个区间个采样一次
        # beta的值不断递增，并且最大值为1
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # 最小概率
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(n):  # 采样n次
            a, b = pri_seg * i, pri_seg * (i + 1)  # 区间端点
            v = np.random.uniform(a, b)  # 在当前区间产生一个随机数
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p  # 概率
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)  # 重要性权重
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        # ti和p分别是叶子节点下标和优先级
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class PrioritizedReplayDQN():
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=2000,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name="IS_weights")
        self.learn_step_counter = 0  # 学习花费的步数
        # 记忆库的大小，用来保存 [s,a,r,s_]，一个状态由两个数字来表示
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.build_net()  # 创建两个神经网络

        self.memory = Memory(self.memory_size)

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
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name="s_")
        self.r = tf.placeholder(tf.float32, [None, ], name="r")
        self.a = tf.placeholder(tf.int32, [None, ], name="a")
        # 初始化权重和偏置
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope("eval_net"):  # 估值网络
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="e1")  # 第1层
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name="e2")  # 第二层

        with tf.variable_scope("target_net"):  # 目标网络
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="t1")
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name="t2")

        with tf.variable_scope("q_target"):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name="Qmax_s_")  # Q真实
            self.q_target = tf.stop_gradient(q_target)  # 对 q_target的反向传播进行截断
        with tf.variable_scope("q_eval"):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)  # 动作的下标
            # 根据索引提取 params 中的元素，构建新的 tensor
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # Q估计
        with tf.variable_scope("loss"):  # 损失函数，本质是 Q 表的更新公式
            # -----------------------------------------------------------------------------#
            self.loss = tf.reduce_mean(
                self.ISWeights * tf.squared_difference(self.q_target, self.q_eval_wrt_a, name="TD_error"))
            # -----------------------------------------------------------------------------#
        with tf.variable_scope("train"):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):  # 存储
        # -----------------------------------------------------------------------------#
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)
        # -----------------------------------------------------------------------------#

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

        # -----------------------------------------------------------------------------#
        batch_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        # -----------------------------------------------------------------------------#

        # 对神经网络进行训练并得到损失
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
                self.ISWeights: ISWeights
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
