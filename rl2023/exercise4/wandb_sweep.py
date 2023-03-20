import sys
import gym
import wandb

from rl2023.exercise4.train_ddpg import BIPEDAL_HPARAMS, WANDB_SWEEP, BIPEDAL_CONFIG, train, NUM_SEEDS_SWEEP
from rl2023.util.result_processing import Run, wandb_data_objects, WANDB_PROJECT

sweep_configuration = {
    'program': 'train_ddpg.py',
    'description': 'Sweep over DDPG model sizes for q4',
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'eval_mean_return'},
    'parameters': {k: {'values': v} for k, v in BIPEDAL_HPARAMS.items()}
}

sweep_configuration['parameters']['seed'] = {'values': list(range(NUM_SEEDS_SWEEP))}

def run_agent():
    CONFIG = BIPEDAL_CONFIG

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
