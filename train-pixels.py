
import sys, os.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'toy_env'))

import numpy as np
import gym
import random
from gym import wrappers
from model import *
import toy_env

from PIL import Image

random.seed()

episode_count = 1000000
steps_per_episode = 1000
step_size = 0
epsilon_steps = 100000
min_epsilon = 0.05
replay_size = 1
batch_size = 1
fc_sizes = [128, 128]
learning_rate = 1e-3
reward_gamma = 0.99
render = False
max_action = True
restore = False

episode_count = 1000000
steps_per_episode = 10000
step_size = 0
epsilon_steps = 1
min_epsilon = 0.0
replay_size = 10
batch_size = 10
fc_sizes = [128]
learning_rate = 1e-3
reward_gamma = 0.99
env_name = 'Pong-v0'
max_action = False

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

env = gym.make(env_name)
num_inputs  = 80*80
num_outputs = 2
print(env.action_space.n)

model = Model(num_inputs, num_outputs, fc_sizes=fc_sizes, learning_rate=learning_rate)

if restore: model.restore()

epsilon = 1.0

experience_replay = []

def add_experience(experience):
  experience_replay.append(experience)
  if len(experience_replay) > replay_size:
    experience_replay.pop(0)

def train_model():
  if len(experience_replay) >= batch_size:
    model.train(random.sample(experience_replay, batch_size), expected_reward, reward_std)

expected_reward = 0
reward_std = 0

prev_state = None

for i in range(episode_count):
  state = env.reset()
  total_reward = 0

  experience = []

  for t in range(steps_per_episode):
    if render: env.render()

    cur_state = prepro(state)
    state = cur_state - prev_state if prev_state is not None else np.zeros(num_inputs)
    # Image.fromarray(state.reshape(80,80)*255).show()
    prev_state = cur_state

    if random.random() < epsilon:
      # action = env.action_space.sample()
      # print action
      action = random.randint(0,1)
    else:
      if max_action: action = model.get_max_action(state)
      else: action = model.get_action(state)
      # print "max action is: " + str(action)

    if epsilon > min_epsilon:
      epsilon -= (1.0 - min_epsilon)/epsilon_steps + 0.00000000001

    # print "Taking action: {}".format(action)
    new_state, reward, done, info = env.step(action+2)
    # if done and t >= 599:
    #   reward += 200

    experience.append((state, action, reward))
    state = new_state
    total_reward += reward

    if step_size != 0 and len(experience) == step_size:
      expected_reward = reward_gamma * expected_reward + (1 - reward_gamma) * total_reward
      reward_std = reward_gamma * reward_std + (1 - reward_gamma) * abs(expected_reward - total_reward)
      add_experience(experience)
      train_model()
      experience = []

    if done:
      expected_reward = reward_gamma * expected_reward + (1 - reward_gamma) * total_reward
      reward_std = reward_gamma * reward_std + (1 - reward_gamma) * abs(expected_reward - total_reward)
      add_experience(experience)
      if i%batch_size == 0:
        train_model()
        with open("output.txt", "a") as outfile:
          outfile.write(("Episode {0:05d} finished after {1:03d} timesteps. Total Reward: {2:03.2f} Expected Reward: {3:03.2f} Reward Std: {4:03.2f} (epsilon: {5:.2f})\n".format(i, t+1, total_reward, expected_reward, reward_std, epsilon)))


      if (i % 100) == 0:
        model.save()

      break

  print("Episode {0:05d} finished after {1:03d} timesteps. Total Reward: {2:03.2f} Expected Reward: {3:03.2f} Reward Std: {4:03.2f} (epsilon: {5:.2f})".format(i, t+1, total_reward, expected_reward, reward_std, epsilon))
