import random
import gym
import gym_drone
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from scores.score_logger import ScoreLogger

# ENV_NAME = "CartPole-v1"

ENV_NAME = "MountainCar-v0"

GAMMA = 0.95
LEARNING_RATE = 0.01

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.tau = 0.05

        self.model = Sequential()
        self.model.add(Dense(24, input_dim=observation_space.shape[0], activation="relu"))
        self.model.add(Dense(48, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space.n))
        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))

        self.target_model = Sequential()
        self.target_model.add(Dense(24, input_dim=observation_space.shape[0], activation="relu"))
        self.target_model.add(Dense(48, activation="relu"))
        self.target_model.add(Dense(24, activation="relu"))
        self.target_model.add(Dense(self.action_space.n))
        self.target_model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
        if np.random.random() < self.exploration_rate:
            return self.action_space.sample()
        prediction = self.model.predict(state)[0]
        print(prediction)
        return np.argmax(prediction)

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            target = self.target_model.predict(state)
            if terminal:
                target[0][action] = reward
            else:
                target[0][action] = reward + max(self.target_model.predict(state_next)[0]) * GAMMA
        self.model.fit(state, target, epochs=1, verbose=0)
        #     q_update = reward
        #     if not terminal:
        #         q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
        #     q_values = self.model.predict(state)
        #     q_values[0][action] = q_update
        #     self.model.fit(state, q_values, verbose=0)
        # self.exploration_rate *= EXPLORATION_DECAY
        # self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)


def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space
    action_space = env.action_space
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space.shape[0]])
        step = 0
        over2 = False
        over0 = False
        over3 = False
        while True:
            step += 1
            action = dqn_solver.act(state)
            env.render()
            state_next, reward, terminal, info = env.step(action)
            # reward = reward if not terminal else -20
            if state_next[0] > -0.2 and not over2:
                over2 = True
                reward = 1
            if state_next[0] > 0 and not over0:
                over0 = True
                reward = 1
            if state_next[0] > 0.2 and not over3:
                over3 = True
                reward = 1
            if state_next[0] >=0.5:
                reward = 20

            # reward = -reward
            state_next = np.reshape(state_next, [1, observation_space.shape[0]])
            dqn_solver.remember(state, action, reward, state_next, terminal)

            dqn_solver.experience_replay()
            dqn_solver.target_train()

            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(reward))
                print("state: " + str(state_next[0,0]))
                # score_logger.add_score(reward, run)
                break


if __name__ == "__main__":
    cartpole()
