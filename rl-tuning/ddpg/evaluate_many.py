from typing import List, Tuple
from pathlib import Path

import gym
from tqdm import tqdm, trange

from .train import play_episode, BIPEDAL_CONFIG
from .agent import DDPG

RENDER = True

CONFIG = BIPEDAL_CONFIG
CONFIG["save_filename"] = "/home/james/School/RL/uoe-rl2023-coursework/rl2023/exercise5/bipedal_q5_latest.pt"


def evaluate(env: gym.Env, config) -> Tuple[
    List[float], List[float]]:
    """
    Execute training of DDPG on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    agent = DDPG(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    try:
        agent.restore(config['save_filename'])
        # agent.save(config['save_filename'])
    except:
        raise ValueError(f"Could not find model to load at {config['save_filename']}")

    eval_returns_all = []
    eval_times_all = []

    eval_returns = 0
    for _ in trange(config["eval_episodes"], leave=False):
        ep_timesteps, episode_return, ep_data = play_episode(
            env,
            agent,
            0,
            train=False,
            explore=False,
            render=RENDER,
            max_steps=config["episode_length"],
            batch_size=config["batch_size"],
        )
        eval_returns += episode_return / config["eval_episodes"]
        eval_times_all.append(ep_timesteps)
        eval_returns_all.append(episode_return)

    return eval_returns, eval_times_all, eval_returns_all


def evaluate_all(savefile_dir=Path("./300runs"), n_runs=500):
    CONFIG["eval_episodes"] = n_runs

    env = gym.make(CONFIG["env"])
    runs = {"name": [], "run": [], "return": [], "time": []}
    files = list(savefile_dir.glob("*.pt"))
    for weights_file in tqdm(files):
        CONFIG["save_filename"] = weights_file.resolve()
        mean_return, eval_times, returns = evaluate(env, CONFIG)
        for i, r, time in zip(range(n_runs), returns, eval_times):
            runs["name"].append(weights_file.stem)
            runs["run"].append(i)
            runs["return"].append(r)
            runs["time"].append(time)

    # import pandas as pd
    # df = pd.DataFrame(runs)
    # df.to_csv(savefile_dir / "eval.csv")

    env.close()


if __name__ == "__main__":
    # evaluate_all()
    evaluate(gym.make(CONFIG["env"]), CONFIG)
