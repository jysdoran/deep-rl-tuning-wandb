import wandb

api = wandb.Api()

# Access attributes directly from the run object or from the W&B App
username = "jysdoran"
project = "rl-coursework"

# TAG = "Monte Carlo"
TAG = "Q-Learning"
# Get all runs in the project
runs = api.runs(f"{username}/{project}", filters={"tags": TAG})

# Add algorithm to config
for run in runs:
    run.config["Algorithm"] = TAG
    run.update()
