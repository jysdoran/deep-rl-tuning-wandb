import os
import gym
import numpy as np
from torch import Tensor
from torch.optim import Adam
from typing import Dict, Iterable
import torch
from torch.autograd import Variable

from ..provided import FCNetwork, Transition, Agent


class DiagGaussian(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        eps = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std * eps


class DDPG(Agent):
    """ DDPG
        :attr critic (FCNetwork): fully connected critic network
        :attr critic_optim (torch.optim): PyTorch optimiser for critic network
        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for actor network
        :attr gamma (float): discount rate gamma
        """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            batch_norm: bool = False,
            **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        # self.actor = Actor(STATE_SIZE, policy_hidden_size, ACTION_SIZE)
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )

        self.actor_target.hard_update(self.actor)
        # self.critic = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)
        # self.critic_target = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau

        # ################################################### #
        # DEFINE A GAUSSIAN THAT WILL BE USED FOR EXPLORATION #
        # ################################################### #
        mean = torch.zeros(ACTION_SIZE)
        std = 0.1 * torch.ones(ACTION_SIZE)
        self.noise = DiagGaussian(mean, std)

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #
        if batch_norm:
            self.actor_batch_norm = torch.nn.BatchNorm1d(STATE_SIZE)
        else:
            self.actor_batch_norm = None

        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )

        # GPU support
        # This only helps if the network is large enough to outweigh the overhead
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for k, model in self.saveables.items():
            if not k.endswith("_optim") and model is not None:
                model.to(self.device)
        if self.actor_batch_norm is not None:
            self.actor_batch_norm.to(self.device)

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        save_dict = self.saveables.copy()
        if self.actor_batch_norm is not None:
            save_dict["actor_batch_norm"] = self.actor_batch_norm
        torch.save({k: v if k.endswith("_optim") else v.to("cpu") for k, v in save_dict.items()}, path)
        return path

    def restore(self, filename: str, dir_path: str = None):
        """Restores PyTorch models from models file given by path

        :param filename (str): filename containing saved models
        :param dir_path (str, optional): path to directory where models file is located
        """

        if dir_path is None:
            dir_path, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dir_path, filename)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

        if "actor_batch_norm" in checkpoint:
            if self.actor_batch_norm is None:
                raise ValueError("Cannot restore batch norm state if not used in agent")
            self.actor_batch_norm.load_state_dict(checkpoint["actor_batch_norm"].state_dict())

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        r = timestep / max_timesteps

        def linear_decay(fraction, start, end):
            if r < fraction:
                return start + r * (end - start) / fraction
            else:
                return end

        def exponential_decay(current, start, end, decay_factor):
            if current > end:
                new_epsilon = np.float_power(decay_factor, r) * start
                return max(new_epsilon, end)
            else:
                return end

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        When explore is False you should select the best action possible (greedy). However, during exploration,
        you should be implementing exporation using the self.noise variable that you should have declared in the __init__.
        Use schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        with torch.no_grad():
            state = Tensor(obs).unsqueeze(0).to(self.device)
            if self.actor_batch_norm is not None:
                self.actor_batch_norm.eval()
                state = self.actor_batch_norm(state)

            action = self.actor(state).squeeze(0).to("cpu")
            if explore:
                action += self.noise.sample()

        return action.clamp(self.lower_action_bound, self.upper_action_bound).numpy()

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your critic and actor networks, target networks with soft
        updates, and return the q_loss and the policy_loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        o_states, actions, next_states, rewards, done = batch

        # Batch normalise states
        if self.actor_batch_norm is not None:
            self.actor_batch_norm.train()
            next_states = self.actor_batch_norm(next_states)
            states = self.actor_batch_norm(o_states)
        else:
            states = o_states

        # Update critic
        target_next_actions = self.actor_target(next_states)
        target_next_q = self.critic_target(torch.cat((next_states, target_next_actions), dim=1))

        current_q = self.critic(torch.cat((states, actions), dim=1))
        q_errors = rewards + self.gamma * (1 - done) * target_next_q - current_q
        critic_loss = q_errors.square().mean()

        self.critic_optim.zero_grad()
        # I think this zero grad should be enough to get rid of the update from the
        # previous actor loss
        critic_loss.backward()
        self.critic_optim.step()

        if self.actor_batch_norm is not None:
            # This is repeated because backwards() erases the involved tensors
            states = self.actor_batch_norm(o_states)

        # Update actor
        predicted_actions = self.actor(states)
        predicted_q = self.critic(torch.cat((states, predicted_actions), dim=1))
        actor_loss = -predicted_q.mean()

        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        # Update target networks
        self.actor_target.soft_update(self.actor, self.tau)
        self.critic_target.soft_update(self.critic, self.tau)

        q_loss = critic_loss.item()
        p_loss = actor_loss.item()
        return {
            "q_loss": q_loss,
            "p_loss": p_loss,
        }
