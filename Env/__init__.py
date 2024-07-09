from gymnasium.envs.registration import register

register(
    id='PendulumFixPos-v0',
    entry_point='Env.envs:PendulumFixPos',
    max_episode_steps=200,
)
