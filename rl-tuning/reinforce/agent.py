from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List, Tuple

from ..provided import FCNetwork, Transition, Agent


class Reinforce(Agent):
    """ The Reinforce Agent

    :attr policy (FCNetwork): fully connected network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for policy network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            learning_rate: float,
            hidden_size: Iterable[int],
            gamma: float,
            **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        pass

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        with torch.no_grad():
            distribution = self.policy(Tensor(obs))

            if explore:
                action = Categorical(distribution).sample().item()
            else:
                action = distribution.argmax().item()

        return action

    def update(
            self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
    ) -> Dict[str, float]:
        """Update function for policy gradients

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """
        ### PUT YOUR CODE HERE ###
        # Compute the discounted returns
        observations = np.array(observations)
        actions = np.array(actions)
        n = len(observations)

        returns = np.array(rewards)
        for i in range(len(returns) - 2, -1, -1):
            returns[i] += self.gamma * returns[i + 1]

        # Compute the loss
        action_probs = self.policy(Tensor(observations))[np.arange(n), actions]
        if not action_probs.all():
            # salt values to avoid log(0)
            # (I think this is caused by policy/categorical non-determinism)
            action_probs = torch.where(action_probs > 0, action_probs, 1e-6)

        loss = (-action_probs.log() * Tensor(returns)).mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        p_loss = loss.item()
        return {"p_loss": p_loss}
