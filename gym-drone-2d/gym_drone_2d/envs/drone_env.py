import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
# import time

MIN = -25.
MAX = 25.
TIME_MAX = 10
DART_RADIUS = 30
# MAX_VELOCITY = np.array([1.,1.])  # ???
MAX_SPEED = 1.
TIMESTEP = 0.5


directions = [np.array([0.,0.]), np.array([1.,0.]), np.array([-1.,0.]), np.array([0.,1.]),
              np.array([0., -1.])]

class DroneEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.min_position = np.array([MIN, MIN])
    self.max_position = np.array([MAX, MAX])

    # self.dart_center = np.array([random.uniform(MIN, MAX), random.uniform(MIN, MAX), 0])

    self.action_space = spaces.Discrete(6)

    # observation space has: muav_pos, distance to center, timestep
    low = np.array([MIN,MIN, 0, 0])
    high = np.array([MAX,MAX, np.linalg.norm(self.max_position - self.min_position), TIME_MAX])

    self.observation_space = spaces.Box(low, high, dtype=np.float32)

    self.state = None
    self.viewer = None

    # self.velocity = np.array([0.,0.])

    self.steps = 0


  # def normalised(self, state):
  #   muav_pos = np.array([0., 0.])
  #   muav_pos[0], muav_pos[1], distance_to_center, steps = state
  #   muav_pos[0] = (muav_pos[0] - (MAX - MIN) / 2 - MIN) / ((MAX - MIN) / 2)
  #   muav_pos[1] = (muav_pos[1] - (MAX - MIN) / 2 - MIN) / ((MAX - MIN) / 2)
  #   # distance_to_center = (muav_pos[1] - np.linalg.norm(self.max_position - self.min_position) / 2) / (np.linalg.norm(self.max_position - self.min_position) / 2)
  #   distance_to_center = np.linalg.norm(muav_pos)
  #   return (muav_pos[0], muav_pos[1], distance_to_center, steps)


  def step(self, action):
    if self.steps >= TIME_MAX:
      # time is over -> end game and give reward -100
      done = True
      reward = -100
      return np.array(self.state), reward, done, {}
    self.steps += 1


    print("step")
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    state = self.state
    muav_pos = np.array([0.,0.])
    muav_pos[0], muav_pos[1], distance_to_center, steps = state

    reward = 0.
    done = False
    new_muav_pos = muav_pos

    if action == 5:
      # dart was thrown - game is over and reward is negative distance to center
      reward = - np.linalg.norm(np.array([0.,0.]) - muav_pos)
      done = True
    else:
      # apply action to muav_pos and check for collisions!
      ...
      new_direction = directions[action]
      # new_muav_pos = muav_pos + (0.5 * self.velocity + 0.5 * new_direction) * TIMESTEP
      new_muav_pos = muav_pos + new_direction * TIMESTEP
      # self.velocity = np.clip((new_muav_pos - muav_pos) / TIMESTEP, -MAX_SPEED, MAX_SPEED)

      if min(new_muav_pos) < MIN or max(new_muav_pos) > MAX:
        print("collision with wall")
        done = True
        reward = -100



    # for i in range(0,11):
    #   if np.linalg.norm(np.array([25.,25.]) - new_muav_pos) < i*2 and not self.intermediateRewards[i]:
    #     print("crossed over intermed " + str(i))
    #     reward += 5
    #     self.intermediateRewards[i] = True
    #   if np.linalg.norm(np.array([25.,25.]) - new_muav_pos) > i*2 and self.intermediateRewards[i]:
    #     print("crossed back over intermed " + str(i))
    #     reward -= 5
    #     self.intermediateRewards[i] = False

    self.state = (new_muav_pos[0], new_muav_pos[1],
                  np.linalg.norm(np.array([0.,0.]) - new_muav_pos), self.steps)
    return np.array(self.state), reward, done, {}


  def reset(self):
    pos = np.array([random.uniform(MIN, MAX), random.uniform(MIN, MAX)])
    self.steps = 0
    # dart_center = np.array([random.uniform(MIN, MAX), random.uniform(MIN, MAX)], 0)
    self.state = (pos[0], pos[1], np.linalg.norm(np.array([0.,0.]) - pos), self.steps)
    # self.intermediateRewards = [False] * 11

    # for i in range(0,11):
    #   if np.linalg.norm(np.array([25.,25.]) - pos) < i*2 and not self.intermediateRewards[i]:
    #     print("crossed over intermed " + str(i))
    #     self.intermediateRewards[i] = True

    return np.array(self.state)


  def render(self, mode='human', close=False):
    # print("current state = " + str(self.state))
    # time.sleep(0)
    screen_width = 600
    screen_height = 600

    scale = 600 / (MAX - MIN)

    cartwidth = 5.0
    cartheight = 5.0

    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)
      l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
      cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      self.carttrans = rendering.Transform()
      cart.add_attr(self.carttrans)
      self.viewer.add_geom(cart)

      dart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
      dart.set_color(1,0,0)
      self.darttrans = rendering.Transform()
      dart.add_attr(self.darttrans)
      self.viewer.add_geom(dart)

    if self.state is None: return None

    x = self.state
    cartx = (x[0] - MIN) * scale # MIDDLE OF CART
    carty = (x[1] - MIN) * scale
    self.carttrans.set_translation(cartx, carty)
    dartx = (0. - MIN) * scale  # MIDDLE OF CART
    darty = (0. - MIN) * scale
    self.darttrans.set_translation(dartx, darty)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None