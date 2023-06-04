CARTPOLE_CONSTANTS = {
    "env": "CartPole-v1",
    "gamma": 0.99,
    "episode_length": 400,
    "max_time": 30 * 60,
    "save_filename": None,
    "algo": None,
}

DQN_CARTPOLE_CONSTANTS = CARTPOLE_CONSTANTS.copy()
DQN_CARTPOLE_CONSTANTS["max_timesteps"] = 25000
DQN_CARTPOLE_CONSTANTS["algo"] = "DQN"

REINFORCE_CARTPOLE_CONSTANTS = CARTPOLE_CONSTANTS.copy()
REINFORCE_CARTPOLE_CONSTANTS["max_timesteps"] = 200000
REINFORCE_CARTPOLE_CONSTANTS["algo"] = "Reinforce"

ACROBOT_CONSTANTS = {
    "env": "Acrobot-v1",
    "gamma": 1.0,
    "episode_length": 1000,
    "max_time": 30 * 60,
    "save_filename": None,
    "algo": None,
}

DQN_ACROBOT_CONSTANTS = ACROBOT_CONSTANTS.copy()
DQN_ACROBOT_CONSTANTS["max_timesteps"] = 100000
DQN_ACROBOT_CONSTANTS["algo"] = "DQN"

REINFORCE_ACROBOT_CONSTANTS = ACROBOT_CONSTANTS.copy()
REINFORCE_ACROBOT_CONSTANTS["max_timesteps"] = 700000
REINFORCE_ACROBOT_CONSTANTS["algo"] = "Reinforce"

PENDULUM_CONSTANTS = {
    "env": "Pendulum-v1",
    "target_return": -300.0,
    "episode_length": 200,
    "max_timesteps": 400000,
    "max_time": 120 * 60,
    "gamma": 0.99,
    "save_filename": "pendulum_latest.pt",
    "algo": "DDPG",
}


BIPEDAL_CONSTANTS = {
    "env": "BipedalWalker-v3",
    "eval_freq": 20000,
    "eval_episodes": 100,
    "target_return": 300.0,
    "episode_length": 1600,
    "max_timesteps": 400000,
    "max_time": 120 * 60,
    "save_filename": "bipedal_latest.pt",
    "algo": "DDPG",
}
