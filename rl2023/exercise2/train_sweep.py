import copy
import pickle

import gym
import numpy as np

from rl2023.util.result_processing import Run, wandb_data_objects
from rl2023.util.hparam_sweeping import generate_hparam_configs

from train_monte_carlo import train as train_mc, CONFIG as MC_CONFIG
from train_q_learning import train as train_q, CONFIG as Q_CONFIG

SWEEP = True
SWEEP_SAVE_RESULTS = True
ALG = "Q"
NUM_SEEDS_SWEEP = 50

MC_HPARAMS = {
    "epsilon": (0.6,),
    "gamma": (0.99, 0.8),
}
Q_HPARAMS = {
    "alpha": (0.05,),
    "epsilon": (0.6,),
    "gamma": (0.99, 0.8),
}
SWEEP_RESULTS_FILE_MC = "MC-Taxi-sweep-results.pkl"
SWEEP_RESULTS_FILE_Q = "Q-Taxi-sweep-results.pkl"

if __name__ == "__main__":
    if ALG == "MC":
        CONFIG = MC_CONFIG
        HPARAMS_SWEEP = MC_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_MC
        train = train_mc
    elif ALG == "Q":
        CONFIG = Q_CONFIG
        HPARAMS_SWEEP = Q_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_Q
        train = train_q
    else:
        raise (ValueError(f"Unknown algorithm {ALG}"))

    env = gym.make(CONFIG["env"])

    if SWEEP and HPARAMS_SWEEP is not None:
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
        results = []
        for config in config_list:
            run = Run(config)
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            run.run_name = hparams_values
            print(f"\nStarting new run...")
            for i in range(NUM_SEEDS_SWEEP):
                print(f"\nTraining iteration: {i + 1}/{NUM_SEEDS_SWEEP}")
                run_save_filename = '--'.join([ALG, run.config["env"], hparams_values, str(i)])
                total_reward, eval_returns, eval_neg, q_table = train(env, run.config)
                run.update(eval_returns, np.arange(len(eval_returns)), run_data={"q_table": q_table})
            results.append(copy.deepcopy(run))
            print(f"Finished run with hyperparameters {hparams_values}. "
                  f"Mean final score: {run.final_return_mean} +- {run.final_return_ste}")

        if SWEEP_SAVE_RESULTS:
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)

    else:
        _ = train(env, CONFIG)

    env.close()
