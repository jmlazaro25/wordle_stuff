from __future__ import annotations

from math import exp
from os import path
from os import makedirs
from collections import deque
from collections import namedtuple
from itertools import pairwise
from random import sample
from random import randint
from random import random as randfloat

import torch
import torch.nn as nn
from numpy import prod

from wordle import Wordle
from wordle import load_vocab
from wordle import ALPH_LEN

from typing import Iterable
from typing import Callable
from typing import Union
from numpy import ndarray
from numpy import int64
training_history = dict[str, tuple[float]]


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'is_terminal')
)


class ReplayMemory():
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Checkpointer():
    def __init__(self, dir: str, model_file: str, optim_file: str) -> None:
        self.dir = dir
        self.model_file = path.join(self.dir, model_file)
        self.optim_file = path.join(self.dir, optim_file)
        self.best_loss = None

        makedirs(self.dir, exist_ok=True)

    def checkpoint(
        self,
        loss: float,
        model: nn.Module,
        optim: torch.optim
    ) -> None:
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            model.eval()
            torch.save(model.state_dict(), self.model_file)
            torch.save(optim.state_dict(), self.optim_file)
            model.train()


class EarlyStopper():
    def __init__(
        self,
        min_delta: float = 0,
        patience: int = 0,
        start_episode: int = 0
    ) -> None:
        self.min_delta = min_delta
        self.patience = patience
        self.start_episode = start_episode
        self.episode_last_improved = 0
        self.best_loss = None
        self.stopped = False

    def stop(self, loss: float, episode_n: int) -> bool:
        if self.best_loss is None or loss <= self.loss - self.min_delta:
            self.loss = loss
            self.episode_last_improved = episode_n
        elif (
            episode_n > start_episode + self.patience
            and episode_n > sef.episode_last_improved + self.patience
        ):
            self.stopped = True
        return self.stopped


ConvPoolConfig = namedtuple(
    'ConvPoolConfig',
    (
        'conv_out',
        'conv_kernel_size',
        'pool_kernel_size',
        'pool_padding',
        'dropout'
    )
)

FCConfig = namedtuple('FCConfig', ('f_out', 'dropout'))


class DQN(nn.Module):
    n_char_channels = 1 + ALPH_LEN
    n_hint_channels = Wordle.n_hint_states
    channels_in = n_char_channels + n_hint_channels
    def __init__(
        self,
        target_len: int,
        vocab: Iterable[str],
        conv_pool_configs: Iterable[ConvPoolConfig],
        fc_configs: list[FCConfig]
    ) -> None:
        super(DQN, self).__init__()
        self.target_len = target_len
        self.input_size = DQN.get_input_size(self.target_len)
        self.vocab = vocab
        self.n_actions = len(self.vocab)
        if self.n_actions != fc_configs[-1].f_out:
            raise ValueError(f'''
                fc_configs[-1].f_out ({fc_configs[-1].f_out}) should equal n_actions ({n_actions})
            ''')

        self.conv_pool_configs = conv_pool_configs
        self.fc_configs = fc_configs

        self.conv_pool_layers = DQN.make_conv_pool_layers(
            self.conv_pool_configs
        )
        self.fcin = DQN.get_fcin(self.input_size, self.conv_pool_layers)
        self.fc_layers = DQN.make_fc_layers(
            self.fcin,
            self.n_actions,
            self.fc_configs
        )

    def forward(self, x):
        x = self.one_hot_state_batch(x)
        x = torch.cat(x, dim=1)
        x = self.conv_pool_layers(x)
        x = x.view(-1, self.fcin)  # -1 for batch size inference
        x = self.fc_layers(x)
        return x

    @staticmethod
    def make_conv_pool_layers(
        conv_pool_configs: Iterable[ConvPoolConfig]
    ) -> nn.Sequential:
        first_config = conv_pool_configs[0]
        conv_pool_layers = [
                nn.Conv2d(
                    DQN.channels_in,
                    first_config.conv_out,
                    kernel_size=first_config.conv_kernel_size,
                    padding='same'
                ),
                nn.ReLU()
        ]
        if first_config.pool_kernel_size:
            conv_pool_layers.append(
                nn.MaxPool2d(
                    kernel_size=first_config.pool_kernel_size,
                    stride=1,
                    padding=first_config.pool_padding
                )
            )
        if first_config.dropout:
            conv_pool_layers.append(nn.Dropout2d(first_config.dropout))

        for config1, config2 in pairwise(conv_pool_configs):
            conv_pool_layers.extend((
                nn.Conv2d(
                    config1.conv_out,
                    config2.conv_out,
                    kernel_size=config2.conv_kernel_size,
                    padding='same'
                ),
                nn.ReLU()
            ))
            if config2.pool_kernel_size:
                conv_pool_layers.append(
                    nn.MaxPool2d(
                        kernel_size=config2.pool_kernel_size,
                        stride=1,
                        padding=config2.pool_padding
                    )
                )
            if config2.dropout:
                conv_pool_layers.append(nn.Dropout2d(config2.dropout))
        return nn.Sequential(*conv_pool_layers)

    @staticmethod
    def get_input_size(target_len):
        return (1, DQN.channels_in, Wordle.max_attempts, target_len)

    @staticmethod
    def get_fcin(
        input_size: tuple[int],
        conv_pool_layers: nn.Sequential
    ) -> int:
        test_tensor = torch.zeros(input_size)
        with torch.no_grad():
            test_out_size = conv_pool_layers(test_tensor).size()
        return prod(test_out_size)

    @staticmethod
    def make_fc_layers(
        fcin: int,
        n_actions: int,
        fc_configs: list[FCConfig]
    ) -> nn.Sequential:
        first_config = fc_configs[0]

        fc_layers = [nn.Linear(fcin, first_config.f_out)]
        if len(fc_configs) > 1:
            fc_layers.append(nn.ReLU())
        if first_config.dropout:
            fc_layers.append(nn.Dropout(first_config.dropout))

        for config1, config2 in pairwise(fc_configs):
            fc_layers.append(nn.Linear(config1.f_out, config2.f_out))
            if config2.f_out != n_actions:
                fc_layers.append(nn.ReLU())
            if config2.dropout:
                fc_layers.append(nn.Dropout(config2.dropout))

        return nn.Sequential(*fc_layers)

    def one_hot_state_batch(
            self,
            state_batch: torch.Tensor
        ) -> tuple:
        batch_size = len(state_batch)
        wordle_grid_size = (Wordle.max_attempts, self.target_len)
        one_hot_chars = torch.zeros(
            (batch_size, DQN.n_char_channels, *wordle_grid_size)
        )
        one_hot_hints = torch.zeros(
            (batch_size, DQN.n_hint_channels, *wordle_grid_size)
        )
        for state_n in range(batch_size):
            reached_attempts_made = False
            for attempt_i in range(Wordle.max_attempts):
                for pos_i in range(self.target_len):
                    space_tuple = (attempt_i, pos_i)
                    char, hint = map(int, state_batch[state_n, :, *space_tuple])
                    if char == Wordle.initial_empty:
                        reached_attempts_made = True
                        break
                    one_hot_chars[state_n, char, *space_tuple] = 1.
                    one_hot_hints[state_n, hint, *space_tuple] = 1.
                if reached_attempts_made:
                    break
        return (one_hot_chars, one_hot_hints)

    def train_model(self, trainer: DQNTrainer, **kwargs) -> training_history:
        target_net = DQN(
            self.target_len,
            self.vocab,
            self.conv_pool_configs,
            self.fc_configs
        )
        target_net.load_state_dict(self.state_dict())
        trainer.set_target_net(target_net)
        return trainer.train(self, **kwargs)


class DQNTrainer():
    def __init__(
        self,
        optimizer,
        checkpointer: Checkpointer = None,
        stopper: EarlyStopper = None,
        device: torch.device = None,
        memory_size: int = 5_000,
        n_episodes: int = 10_000,
        batch_size: int = 128,
        discount: float = 0.99,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        update_rate: float = 1e-4,
        plot_freq: int = 50,
    ) -> None:
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.stopper = stopper
        self.device = device
        self.memory = ReplayMemory(memory_size)
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.discount = discount
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.update_rate = update_rate
        self.plot_freq= plot_freq

        self.step_n = 0
        self.batch_n = 0

        self.target_net = None # set in DQN.train w/ DQNTrainer.set_target_net

    def set_target_net(self, target_net: DQN) -> None:
        self.target_net = target_net
        self.target_net.to(self.device)

    def select_action(
        self,
        policy_net: DQN,
        state: torch.Tensor
    ) -> torch.Tensor:
        self.step_n += 1
        eps_threshold = (
            (self.eps_start - self.eps_end)
            * exp(-1. * self.step_n / self.eps_decay)
            + self.eps_end
        )
        if randfloat() > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[randint(0, policy_net.n_actions - 1)]],
                device=self.device,
                dtype=torch.long
            )

    @staticmethod
    def get_reward(target_len, state, status, guess_n) -> float:
        turn_value = 10.
        reward = -turn_value
        reward += sum(
            1./target_len for hint in state[Wordle.hint_channel, guess_n - 1]
            if hint == Wordle.correct
        )
        reward += Wordle.max_attempts * turn_value * int(status == Wordle.won)
        return reward / target_len

    def optimize_model(self, policy_net) -> Union[None, float]:
        """ Return Huber loss """
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*transitions))

        non_terminal_mask = torch.logical_not(
            torch.tensor(batch.is_terminal, device=self.device, dtype=torch.bool)
        )
        non_terminal_next_states = [
            s for s, nt in zip(batch.next_state, non_terminal_mask) if nt
        ]
        #if not non_terminal_next_states:
        #    return None
        non_terminal_next_states = torch.cat(non_terminal_next_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_terminal_mask] = self.target_net(
                non_terminal_next_states
            ).max(1).values
        # Compute the expected Q values
        expected_state_action_values = torch.where(
            non_terminal_mask,
            reward_batch + next_state_values * self.discount,
            reward_batch
        )

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        self.optimizer.step()

        return float(loss)

    def train(
        self,
        policy_net: DQN,
        n_episodes: int = 0,
        plot_training: Callable = None
    ) -> training_history:
        if n_episodes == 0:
            n_episodes = self.n_episodes

        self.episode_rewards = []
        self.losses = []
        for episode_n in range(n_episodes):
            self.episode_n = episode_n # make accisible to plot_training
            if self.stopper.stopped:
                print(f'Early stopping after {episode_n} episodes')
                break
            wordle = Wordle(policy_net.vocab)
            state = torch.tensor(
                wordle.state,
                dtype=torch.uint8,
                device=self.device
            ).unsqueeze(0)
            status = Wordle.ongoing
            episode_reward = 0
            while status == Wordle.ongoing:
                action = self.select_action(policy_net, state)
                next_state = wordle.guess(policy_net.vocab[action])
                status = wordle.check_state()
                reward = DQNTrainer.get_reward(
                    policy_net.target_len,
                    next_state,
                    status,
                    wordle.attempts_made
                )
                episode_reward += reward
                next_state = torch.tensor(
                    next_state,
                    dtype=torch.uint8,
                    device=self.device
                ).unsqueeze(0)
                reward = torch.tensor([reward], device=self.device)

                # Store the transition in memory
                self.memory.push(
                    state,
                    action,
                    next_state,
                    reward,
                    status != Wordle.ongoing
                )
                state = next_state

                # Optimize policy network and checkpoint/stop as needed
                loss = self.optimize_model(policy_net)
                if loss:
                    self.losses.append(loss)
                    self.batch_n += 1
                    self.checkpointer.checkpoint(
                        loss,
                        policy_net,
                        self.optimizer
                    )
                if self.stopper.stop(loss, episode_n):
                    break

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                policy_net_state_dict = policy_net.state_dict()
                target_net_state_dict = self.target_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (
                        policy_net_state_dict[key] * self.update_rate
                        + target_net_state_dict[key] * (1 - self.update_rate)
                    )
                self.target_net.load_state_dict(target_net_state_dict)

                if status != Wordle.ongoing:
                    self.episode_rewards.append(episode_reward)
                    if plot_training and self.plot_freq:
                        plot_training(self)

        return {
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }
