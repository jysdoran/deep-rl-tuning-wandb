import sys
import gym
import wandb

from .train import BIPEDAL_CONFIG, BIPEDAL_CONSTANTS, train, NUM_SEEDS_SWEEP
from ..result_processing import WANDB_PROJECT

HPARAMS = {
    "policy_learning_rate": [1e-3],
    "critic_learning_rate": [5e-4],
    "critic_hidden_size": [[1024, 512]],
    "policy_hidden_size": [[256, 128]],
    "gamma": [0.98, 0.97, 0.96],
    "tau": [0.1, 0.05, 0.01],
    "batch_size": [128],
    "buffer_capacity": [int(1e6)],
    "seed": list(range(NUM_SEEDS_SWEEP))
}

sweep_configuration = {
    'program': 'train.py',
    'description': 'Sweep over DDPG model parameters for q5',
    'method': 'grid',
    # 'early_terminate': {'type': 'hyperband', 'min_iter': 10},
    'metric': {'goal': 'maximize', 'name': 'eval_mean_return'},
    'parameters': {k: {'values': v} for k, v in HPARAMS.items()}
}


def run_agent():
    # Do not include any of the
    CONFIG = BIPEDAL_CONSTANTS
    env_0 = gym.make(CONFIG["env"])
    env = env_0

    eval_returns, eval_timesteps, times, run_data = train(env, CONFIG)

    run_data.run.finish()


if __name__ == '__main__':
    # assert WANDB_SWEEP, "WANDB_SWEEP must be True to use wandb sweep"
    if len(sys.argv) > 1:
        sweep_id = sys.argv[1]
        wandb.agent(sweep_id, function=run_agent)
    else:
        sweep_id = wandb.sweep(sweep_configuration, project=WANDB_PROJECT)
