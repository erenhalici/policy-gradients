
import gym
from gym import spaces
import numpy as np

class Toy(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Discrete(3)
    self._sequence = [0, 1, 0]

  def _reset(self):
    self._state = 0
    self._failed = False
    return self.state_vector(self._state)

  def _step(self, a):
    if a != self._sequence[self._state]:
      self._failed = True

    self._state += 1

    if self._state == 3:
      terminal = True
      if self._failed:
        reward = -1
      else:
        reward = 1
    else:
      terminal = False
      reward = 0

    return self.state_vector(self._state), reward, terminal, {}

  def _render(self, mode='human', close=False):
    if close:
      return

    if mode == 'human':
      if self._state == 3:
        print "Game Ended! Failed: {}".format(self._failed)
      else:
        print "State: {}, Expected Input: {}, Failed: {}".format(self._state, self._sequence[self._state], self._failed)
    # elif mode == 'rgb_array':
      # return img



  def state_vector(self, state):
    state_vector = np.zeros(3)
    if state < 3:
      state_vector[self._state] = 1
    return state_vector

  @property
  def _n_actions(self):
      return 2
