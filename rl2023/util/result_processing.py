import numpy as np
import wandb
from typing import Dict, List, Tuple
from collections import defaultdict

WANDB_MODE = ("disabled", "online")[1]
WANDB_PROJECT = "rl-coursework-q5"


def wandb_data_objects(config, project=WANDB_PROJECT):
    # wrappers for lists and dict that log to wandb on append
    class WandBList(list):
        def __init__(self, name, buffer, send="time", step="timesteps_elapsed"):
            self.name = name
            self.step = step
            self.send = send
            self.buffer = buffer
            super().__init__()

        @staticmethod
        def process(x):
            if isinstance(x, list):
                return np.mean(x)
            return x

        def append(self, x):
            super().append(x)
            self.buffer[self.name] = x
            if self.name == self.send:
                wandb.log({k: self.process(v) for k, v in self.buffer.items()}, step=self.buffer[self.step])
                self.buffer.clear()

        def extend(self, x):
            super().extend(x)
            self.buffer.setdefault(self.name, []).extend(x)
            if self.name == self.send:
                wandb.log(self.buffer, step=self.buffer[self.step])
                self.buffer.clear()

    class WandBRunData(dict):
        def __init__(self, run, step="train_ep_timesteps", send="train_ep_timesteps"):
            self.run = run
            self.step = step
            self.send = send
            self.buffer = {}
            super().__init__()

        def __getitem__(self, key):
            return super().setdefault(key, WandBList(key, self.buffer, send=self.send, step=self.step))

    run = wandb.init(project=project, mode=WANDB_MODE, reinit=True, config=config)
    wandb.define_metric("eval_mean_return", summary="max")
    eval_buffer = {}
    eval_returns = WandBList("eval_mean_return", eval_buffer)
    eval_timesteps = WandBList("timesteps_elapsed", eval_buffer)
    eval_times = WandBList("time", eval_buffer)

    run_data = WandBRunData(run)

    return eval_returns, eval_timesteps, eval_times, run_data


class Run:

    def __init__(self, config: Dict, tags: List[str] = None):
        self._config = config
        self._run_name = None

        self._final_returns = []
        self._train_times = []
        self._run_data = []
        self._agent_weights_filenames = []

        self._run_ids = []
        self._all_eval_timesteps = []
        self._all_returns = []

        self._tags = tags if tags is not None else []

    def update(self, eval_returns, eval_timesteps, times=None, run_data=None):
        run_id = len(self._run_ids)
        if run_data is not None:
            run_data.run.finish()

        # with wandb.init(project=WANDB_PROJECT, mode=WANDB_MODE, config=self._config, tags=self._tags,
        #                 reinit=True) as run:
        #     run.name = self._run_name
        #     run.config["run_id"] = run_id
        #
        #     if "train_ep_timesteps" in run_data:
        #         q_loss = np.array(run_data["q_loss"])
        #         p_loss = np.array(run_data["p_loss"]) if "p_loss" in run_data else None
        #         prev_ep_step = 0
        #         for ep_step, ep_return in zip(run_data["train_ep_timesteps"], run_data["train_ep_returns"]):
        #             metrics = {"train/episode/Episode Return": ep_return,
        #                        "Train Episode Mean Critic Loss": q_loss[prev_ep_step:ep_step].mean()}
        #             if p_loss is not None:
        #                 metrics["Train Episode Mean Actor Loss"] = p_loss[prev_ep_step:ep_step].mean()
        #             wandb.log(
        #                 metrics,
        #                 step=ep_step)
        #             prev_ep_step = ep_step
        #
        #     print(eval_returns, eval_timesteps)
        #
        #     if times is not None:
        #         for step, mean_return, time in zip(eval_timesteps, eval_returns, times):
        #             wandb.log({"Mean Eval Return": mean_return, "Time": time}, step=step)
        #     else:
        #         for step, mean_return in zip(eval_timesteps, eval_returns):
        #             wandb.log({"Mean Eval Return": mean_return}, step=step)

        self._run_ids.append(run_id)
        if self._config['save_filename'] is not None:
            self._agent_weights_filenames.append(self._config['save_filename'])
            self._config['save_filename'] = None

        self._all_eval_timesteps.append(list(eval_timesteps))
        self._all_returns.append(list(eval_returns))
        self._final_returns.append(eval_returns[-1])
        if times is not None:
            self._train_times.append(list(times)[-1])
        if run_data is not None:
            self._run_data.append({k: list(v) for k, v in run_data.items()})

    def set_save_filename(self, filename):
        if self._config["save_filename"] is not None:
            print(f"Warning: Save filename already set in config. Overwriting to {filename}.")

        self._config['save_filename'] = f"{filename}.pt"

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, name):
        self._run_name = name

    @property
    def final_return_mean(self) -> float:
        final_returns = np.array(self._final_returns)
        return final_returns.mean()

    @property
    def final_return_ste(self) -> float:
        final_returns = np.array(self._final_returns)
        return np.std(final_returns, ddof=1) / np.sqrt(np.size(final_returns))

    @property
    def final_return_iqm(self) -> float:
        final_returns = np.array(self.final_returns)
        q1 = np.percentile(final_returns, 25)
        q3 = np.percentile(final_returns, 75)
        trimmed_ids = np.nonzero(np.logical_and(final_returns >= q1, final_returns <= q3))
        trimmed_returns = final_returns[trimmed_ids]
        return trimmed_returns.mean()

    @property
    def final_returns(self) -> np.ndarray:
        return np.array(self._final_returns)

    @property
    def train_times(self) -> np.ndarray:
        return np.array(self._train_times)

    @property
    def config(self):
        return self._config

    @property
    def run_ids(self) -> List[int]:
        return self._run_ids

    @property
    def agent_weights_filenames(self) -> List[str]:
        return self._agent_weights_filenames

    @property
    def run_data(self) -> List[Dict]:
        return self._run_data

    @property
    def all_eval_timesteps(self) -> np.ndarray:
        return np.array(self._all_eval_timesteps)

    @property
    def all_returns(self) -> np.ndarray:
        return np.array(self._all_returns)


# The helper functions below are provided to help you process the results of your runs.

def rank_runs(runs: List[Run]):
    """Sorts runs by mean final return, highest to lowest."""

    return sorted(runs, key=lambda x: x.final_return_mean, reverse=True)


def get_best_saved_run(runs: List[Run]) -> Tuple[Run, str]:
    """Returns the run with the highest mean final return and the filename of the saved weights of its highest scoring
    seed, if it exists."""

    ranked_runs = rank_runs(runs)
    best_run = ranked_runs[0]

    if best_run.agent_weights_filenames:
        best_run_id = np.argmax(best_run.final_returns)
        return best_run, best_run.agent_weights_filenames[best_run_id]
    else:
        raise ValueError(f"No saved runs found for highest mean final returns run {best_run.run_name}.")
