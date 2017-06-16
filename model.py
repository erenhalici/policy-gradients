
import numpy as np
import tensorflow as tf
import random

class Model(object):
  def __init__(self, width, height, channels, num_outputs, filter_counts=[32, 64, 128], fc_sizes=[128, 128], gamma=0.995, learning_rate=1e-4):
    self._gamma = gamma

    self._num_outputs = num_outputs

    self._states = tf.placeholder(tf.float32, [None, width, height, channels])
    self._actions = tf.placeholder(tf.float32, [None, num_outputs])
    self._rewards = tf.placeholder(tf.float32, [None, num_outputs])

    last_h = self._states
    # last_size = num_inputs

    last_filter_count = channels

    # for count in filter_counts:
    #   h_conv_1 = self.conv_layer(last_h, last_filter_count, count)
    #   h_conv_2 = self.conv_layer(h_conv_1, count, count)
    #   last_h = self.max_pool_2x2(h_conv_2)

    #   width = int(width/2) - 2
    #   height = int(height/2) - 2
    #   last_filter_count = count
    for count in filter_counts:
      h_conv_1 = self.conv_layer(last_h, last_filter_count, count)
      last_h = self.max_pool_2x2(h_conv_1)

      width = int((width-2)/2)
      height = int((height-2)/2)
      last_filter_count = count


    last_size = width * height * last_filter_count
    last_h = tf.reshape(last_h, [-1, last_size])

    for size in fc_sizes:
      W = self.weight_variable([last_size, size])
      b = self.bias_variable([size])
      last_h = tf.nn.relu(tf.matmul(last_h, W) + b)
      last_size = size

    W = self.weight_variable([last_size, self._num_outputs])
    b = self.bias_variable([self._num_outputs])

    self._action_distribution = tf.nn.softmax(tf.nn.relu(tf.matmul(last_h, W) + b))
    self._max_action = tf.argmax(self._action_distribution, 1)

    self._errors = -tf.log(tf.clip_by_value(self._action_distribution, 1e-15, 1.0-1e-15)) * self._rewards
    loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self._action_distribution, 1e-15, 1.0-1e-15)) * self._rewards)
    # loss = self._action_distribution * self._actions

    # self._train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4).minimize(loss)
    self._train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99).minimize(loss)

    self._saver = tf.train.Saver()
    config = tf.ConfigProto(
      device_count = {'GPU': 0}
    )
    self._sess = tf.Session(config=config)
    self._sess.run(tf.global_variables_initializer())

  def get_max_action(self, state):
    [dist, act] = self._sess.run([self._action_distribution, self._max_action], feed_dict={self._states: [state]})
    # print dist[0]
    return act[0]

  def get_action(self, state):
    dist = self._sess.run(self._action_distribution, feed_dict={self._states: [state]})
    # print(dist)
    r = random.random()

    if r > 0.99999: return len(dist[0])-1

    s = 0
    i = -1

    while r >= s:
      i += 1
      s += dist[0][i]

    return i

  def train(self, experience, expected_reward=0.0, reward_std=1.0):
    states, actions, rewards = self.reward_reduced_experience(experience)
    # print rewards
    # print self._sess.run(self._action_distribution, feed_dict={self._states: states})
    # print self._sess.run(self._errors, feed_dict={self._states: states, self._actions: actions, self._rewards: rewards})
    # expected_reward += abs(expected_reward) * 0.01
    expected_reward = np.mean(rewards)
    reward_std = np.std(rewards)
    print expected_reward, reward_std
    self._sess.run(self._train_step, feed_dict={self._states: states, self._actions: actions, self._rewards: (rewards - np.float64(expected_reward))/reward_std})
    # self._sess.run(self._train_step, feed_dict={self._states: states, self._actions: actions, self._rewards: rewards})
    # print self._sess.run(self._action_distribution, feed_dict={self._states: states})
    # print self._sess.run(self._errors, feed_dict={self._states: states, self._actions: actions, self._rewards: rewards})

  def reward_reduced_experience(self, experience_replay):
    states = []
    actions = []
    rewards = []

    reduced_reward = 0
    for experience in experience_replay:
      for (state, action, reward) in reversed(experience):
        reduced_reward += reward
        reward_vector = np.zeros(self._num_outputs)
        reward_vector[action] = reduced_reward
        rewards.append(reward_vector)
        reduced_reward *= self._gamma

        states.append(state)

        action_vector = np.zeros(self._num_outputs)
        action_vector[action] = 1
        actions.append(action_vector)

    # return states, actions, rewards
    return states, actions, rewards

  def save(self, file='model.ckpt'):
    self._saver.save(self._sess, file)

  def restore(self, file='model.ckpt'):
    self._saver.restore(self._sess, file)

  @property
  def states(self):
    return self._states
  @property
  def actions(self):
    return self._actions
  @property
  def rewards(self):
    return self._rewards
  @property
  def action_distribution(self):
    return self._action_distribution
  @property
  def max_action(self):
    return self._max_action

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

  def conv_layer(self, input_layer, input_channes, output_channels):
    W_conv = self.weight_variable([3, 3, input_channes, output_channels])
    b_conv = self.bias_variable([output_channels])
    return tf.nn.relu(self.conv2d(input_layer, W_conv) + b_conv)

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
