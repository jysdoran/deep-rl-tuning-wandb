from pathlib import Path

import gym
import wandb

from train_monte_carlo import train as train_mc, CONFIG as CONFIG_MC
from train_q_learning import train as train_q, CONFIG as CONFIG_Q

log_dir = Path("./runs")

WANDB_MODE = ["disabled", "online"][0]


def param_string(param_set):
    return "_".join([f"{k}{v}" for k, v in param_set.items()])


def all_param_combinations(param_table):
    param_sets = [{}]
    for name, vals in param_table.items():
        new_param_sets = []
        for v in vals:
            for param_set in param_sets:
                new_param_set = param_set.copy()
                new_param_set[name] = v
                new_param_sets.append(new_param_set)
                param_set[name] = v
        param_sets = new_param_sets

    return param_sets


def grid_search(param_table, train_fn, config_base, log_group="", run_id=0):
    param_sets = all_param_combinations(param_table)

    env = gym.make(config_base["env"])
    for param_set in param_sets:
        with wandb.init(project="rl-coursework", reinit=True, config=param_set, tags=["Q2", log_group],
                        mode=WANDB_MODE) as run:
            # writer = SummaryWriter(log_dir=log_dir / log_group / str(param_set) / str(run_id))
            run.config["repeat"] = run_id
            run_config = config_base.copy()
            run_config.update(param_set)
            total_reward, mean_returns, negative_returns, q_table = train_fn(env, run_config)
            for i, (mean, negative) in enumerate(zip(mean_returns, negative_returns)):
                run.log({"Mean Return": mean, "Negative Return": negative}, step=i * run_config["eval_freq"])
                # writer.add_scalar("return_mean", mean, i * run_config["eval_freq"])


def grid_search_mc(i=0):
    param_table = {
        "epsilon": (0.6,),
        "gamma": (0.99, 0.8),
    }

    grid_search(param_table, train_mc, CONFIG_MC, "Monte Carlo", i)


def grid_search_q(i=0):
    param_table = {
        "alpha": (0.05,),
        "epsilon": (0.6,),
        "gamma": (0.99, 0.8),
    }

    grid_search(param_table, train_q, CONFIG_Q, "Q-Learning", i)


def main():
    repeats = 10
    for i in range(repeats):
        grid_search_mc(i)
        grid_search_q(i)


if __name__ == "__main__":
    main()
