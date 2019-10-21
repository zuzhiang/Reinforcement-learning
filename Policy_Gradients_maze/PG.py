import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient():
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.95,
                 output_graph=False,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay  # reward递减率

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        # 以下是神经网络所有的输入
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name="state")
        self.value = tf.placeholder(tf.float32, [None, ], name="value")
        self.action = tf.placeholder(tf.int32, [None, ], name="action")
        # 初始化权重和偏置

        layer1 = tf.layers.dense(
            inputs=self.state,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., 0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name="layer1")
        action_values = tf.layers.dense(
            inputs=layer1,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0., 0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name="layer2")
        self.action_probs = tf.nn.softmax(action_values,name="act_prob")

        with tf.variable_scope("loss"):  # 损失函数，本质是 Q 表的更新公式
            neg_log_probs = tf.reduce_sum(-tf.log(self.action_probs)* tf.one_hot(self.action, self.n_actions),axis=1)
            self.loss = tf.reduce_mean(neg_log_probs*self.value)
        with tf.variable_scope("train"):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, state, action, reward):  # 存储
        self.ep_obs.append(state)
        self.ep_as.append(action)
        self.ep_rs.append(reward)

    def choose_action(self, observation):  # 选择行动
        prob_weights = self.sess.run(self.action_probs, feed_dict={self.state: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def learn(self):
        discounted_ep_rs = self.discount()
        self.sess.run(self._train_op, feed_dict={
            self.state: np.vstack(self.ep_obs),
            self.action: np.array(self.ep_as),
            self.value: discounted_ep_rs
        })
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs

    def discount(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float32)  # 产生一个和 self.ep_rs 维度相同的全零数组
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 对每个回合的 reward 进行标准化处理
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs