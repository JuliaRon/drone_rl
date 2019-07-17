import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import time

MIN = 0
MAX = 20
TIME_MAX = 300
DART_RADIUS = 30
MAX_VELOCITY = np.array([1,1,1])  # ???
MAX_SPEED = 1
DRAG_COEFFICIENT = 0.3 # possibly way smaller?
TIMESTEP = 0.2

directions = []
for i in range(-1,2):
  for j in range(-1,2):
    for k in range(-1,2):
      x = np.array([i, j, k])
      if i == 0 and j == 0 and k == 0:
        directions.append(x)
        continue
      directions.append(x / np.linalg.norm(x) * np.linalg.norm(MAX_VELOCITY))

class DroneEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.min_position = np.array([MIN, MIN, MIN])
    self.max_position = np.array([MAX, MAX, MAX])
    # self.max_speed = 1
    # self.dart_center = np.array([random.uniform(MIN, MAX), random.uniform(MIN, MAX), 0])

    self.action_space = spaces.Discrete(28) # 0 to 26: directions to move to, 27: throw dart

    # observation space has: muav_pos, muav_velocity, dart_center, distance to center, time left
    # low = np.array([self.min_position, np.array([0,0,0]), self.min_position, 0, 0])
    # high = np.array([self.max_position, MAX_VELOCITY, self.max_position, np.linalg.norm(self.max_position - self.min_position), TIME_MAX])
    low = np.array([MIN,MIN,MIN, 0, 0, 0, MIN,MIN,MIN, 0, 0])
    high = np.array([MAX,MAX,MAX, 1,1,1, MAX,MAX,MAX, np.linalg.norm(self.max_position - self.min_position), TIME_MAX])

    self.observation_space = spaces.Box(low, high, dtype=np.float32)

    self.state = None
    self.viewer = None

  def step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    state = self.state
    muav_pos = np.array([0.,0.,0.])
    muav_velocity = np.array([0., 0., 0.])
    dart_center = np.array([0., 0., 0.])
    # print("state is: " + str(state))
    muav_pos[0], muav_pos[1], muav_pos[2], muav_velocity[0], muav_velocity[1], muav_velocity[2], dart_center[0], dart_center[1], dart_center[2], distance_to_center, time_left = state

    reward = 0
    done = False
    time_left -= 1
    new_muav_pos = muav_pos
    new_muav_velocity = muav_velocity

    # resistance on current muav_velocity:
    # drag equation: drag force = 1/2 * fluid density * speed of object ^ 2 * drag coefficient * cross sectional area
    # i assume drag coefficient is constant


    if action == 27:
      # dart was thrown - game is over
      dart_pos = np.append(muav_pos[0:2], 0) # i dont care about z coord here (z = 0)
      reward = int(DART_RADIUS - np.linalg.norm(dart_center - dart_pos))
      done = True
    else:
      # apply action to muav_pos and check for collisions!
      ...
      new_direction = directions[action]
      # print("next step is: " + str((0.5 * muav_velocity + 0.5 * new_direction) * TIMESTEP))
      new_muav_pos = muav_pos + (0.5 * muav_velocity + 0.5 * new_direction) * TIMESTEP
      new_muav_velocity = np.clip((new_muav_pos - muav_pos) / TIMESTEP, -MAX_SPEED, MAX_SPEED)

      if min(new_muav_pos) < MIN or max(new_muav_pos) > MAX:
        print("collision with wall")
        done = True
        reward = -1

      # TODO: check for collision!

    if time_left == 0:
      done = True

    self.state = (new_muav_pos[0], new_muav_pos[1], new_muav_pos[2], new_muav_velocity[0], new_muav_velocity[1], new_muav_velocity[2], dart_center[0], dart_center[1], dart_center[2], np.linalg.norm(dart_center - new_muav_pos), time_left)
    return np.array(self.state), reward, done, {}


  def reset(self):
    # pos = np.array([random.uniform(MIN, MAX), random.uniform(MIN, MAX), random.uniform(MIN, MAX)])
    # dart_center = np.array([random.uniform(MIN, MAX), random.uniform(MIN, MAX)], 0)
    self.state = (random.uniform(MIN, MAX), random.uniform(MIN, MAX), random.uniform(MIN, MAX), 0,0,0, random.uniform(MIN, MAX), random.uniform(MIN, MAX), 0, MAX, TIME_MAX)
    return np.array(self.state)


  # def render(self, mode='human', close=False):
  #   # print("current state = " + str(self.state))
  #   screen_width = 600
  #   screen_height = 400
  #   carty = 20
  #
  #   world_width = 2.4 * 2
  #   scale = screen_width / world_width
  #   cartwidth = 50.0
  #   cartheight = 50.0
  #
  #   if self.viewer is None:
  #     from gym.envs.classic_control import rendering
  #     self.viewer = rendering.Viewer(screen_width, screen_height)
  #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
  #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
  #     self.carttrans = rendering.Transform()
  #     cart.add_attr(self.carttrans)
  #     self.viewer.add_geom(cart)
  #
  #   if self.state is None: return None
  #
  #   x = self.state
  #   cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
  #   self.carttrans.set_translation(cartx, carty)
  #
  #   return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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
    dartx = (x[6] - MIN) * scale  # MIDDLE OF CART
    darty = (x[7] - MIN) * scale
    self.darttrans.set_translation(dartx, darty)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None