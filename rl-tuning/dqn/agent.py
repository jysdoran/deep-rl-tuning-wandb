from copy import deepcopy
import gym
import numpy as np
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List, Tuple

from ..provided import FCNetwork, Transition, Agent


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_strategy: str = "constant",
        epsilon_decay: float = None,
        exploration_fraction: float = None,
        **kwargs,
        ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        :param epsilon_start (float): initial value of epsilon for epsilon-greedy action selection
        :param epsilon_min (float): minimum value of epsilon for epsilon-greedy action selection
        :param epsilon_decay (float, optional): decay rate of epsilon for epsilon-greedy action. If not specified,
                                                epsilon will be decayed linearly from epsilon_start to epsilon_min.
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
            )

        self.critics_target = deepcopy(self.critics_net)
        # Disable gradient calculations for target network
        self.critics_target.requires_grad_(False)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
            )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min

        self.epsilon_decay_strategy = epsilon_decay_strategy
        if epsilon_decay_strategy == "constant":
            assert epsilon_decay is None, "epsilon_decay should be None for epsilon_decay_strategy == 'constant'"
            assert exploration_fraction is None, "exploration_fraction should be None for epsilon_decay_strategy == 'constant'"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = None
        elif self.epsilon_decay_strategy == "linear":
            assert epsilon_decay is None, "epsilon_decay is only set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is not None, "exploration_fraction must be set for epsilon_decay_strategy='linear'"
            assert exploration_fraction > 0, "exploration_fraction must be positive"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = exploration_fraction
        elif self.epsilon_decay_strategy == "exponential":
            assert epsilon_decay is not None, "epsilon_decay must be set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is None, "exploration_fraction is only set for epsilon_decay_strategy='linear'"
            self.epsilon_exponential_decay_factor = epsilon_decay
            self.exploration_fraction = None
        else:
            raise ValueError("epsilon_decay_strategy must be either 'linear' or 'exponential'")
        # ######################################### #
        self.saveables.update(
            {
                "critics_net"   : self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim"  : self.critics_optim,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        ** Implement both epsilon_linear_decay() and epsilon_exponential_decay() functions **
        ** You may modify the signature of these functions if you wish to pass additional arguments **

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        r = timestep / max_timestep
        def epsilon_linear_decay(*args, **kwargs):
            ### PUT YOUR CODE HERE ###
            # r = timestep / max_timestep
            if r < self.exploration_fraction:
                return self.epsilon_start + r * (self.epsilon_min - self.epsilon_start) / self.exploration_fraction
            else:
                return self.epsilon_min

        def epsilon_exponential_decay(*args, **kwargs):
            ### PUT YOUR CODE HERE ###
            if self.epsilon > self.epsilon_min:
                new_epsilon = np.float_power(self.epsilon_exponential_decay_factor, r) * self.epsilon_start
                return max(new_epsilon, self.epsilon_min)
            else:
                return self.epsilon_min

        if self.epsilon_decay_strategy == "constant":
            pass
        elif self.epsilon_decay_strategy == "linear":
            # linear decay
            ### PUT YOUR CODE HERE ###
            self.epsilon = epsilon_linear_decay()
        elif self.epsilon_decay_strategy == "exponential":
            # exponential decay
            ### PUT YOUR CODE HERE ###
            self.epsilon = epsilon_exponential_decay()
        else:
            raise ValueError("epsilon_decay_strategy must be either 'constant', 'linear' or 'exponential'")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        if explore and np.random.random_sample() < self.epsilon:
            # Random
            action = self.action_space.sample()
        else:
            # Greedy
            with torch.no_grad():
                action = self.critics_net(Tensor(obs).unsqueeze(0)).argmax(axis=-1).item()
            # action = self.q_max(obs[np.newaxis, :], self.critics_net)[1].squeeze(0).item()
        return action

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        states, actions, next_states, rewards, done = batch

        # Compute the target Q value

        # target_vals, target_argmaxes = self.q_max(next_states, self.critics_target)
        # target_q_best = target_vals[range(self.batch_size), target_argmaxes]
        target_vals = self.critics_target(next_states)
        target_q_max = target_vals.max(axis=-1).values

        # Calculate the loss
        net_vals = self.critics_net(states)
        # net_q_taken = net_vals[range(self.batch_size), actions.squeeze(-1).numpy()]
        net_q_taken = net_vals[range(self.batch_size), actions.squeeze(-1).numpy()]


        errors = rewards.squeeze(-1) + self.gamma * (1 - done.squeeze(-1)) * target_q_max - net_q_taken
        loss_mse = errors.square().mean()

        self.critics_optim.zero_grad()
        loss_mse.backward()
        self.critics_optim.step()

        q_loss = loss_mse.item()

        # Update target network every target_update_freq times
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.critics_target.hard_update(self.critics_net)

        return {"q_loss": q_loss}


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
