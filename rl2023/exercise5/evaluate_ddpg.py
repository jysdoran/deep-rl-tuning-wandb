import gym
from typing import List, Tuple

from rl2023.exercise4.agents import DDPG
from rl2023.exercise4.evaluate_ddpg import evaluate
from rl2023.exercise5.train_ddpg \
    import BIPEDAL_CONFIG

RENDER = True

CONFIG = BIPEDAL_CONFIG
CONFIG["save_filename"] = "/home/james/School/RL/uoe-rl2023-coursework/rl2023/exercise5/bipedal_q5_latest.pt"

if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
