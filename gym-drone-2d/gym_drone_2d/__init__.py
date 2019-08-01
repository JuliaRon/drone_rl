from gym.envs.registration import register

register(
    id='drone-2d-v0',
    entry_point='gym_drone_2d.envs:DroneEnv',
    max_episode_steps=300,
)