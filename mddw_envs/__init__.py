from gymnasium.envs.registration import register

register(
     id="mddw_envs/Env1a_2024-v0",
     entry_point="mddw_envs.envs:Env1a_2024",
)

register(
     id="mddw_envs/Env1b_2024-v0",
     entry_point="mddw_envs.envs:Env1b_2024",
)
