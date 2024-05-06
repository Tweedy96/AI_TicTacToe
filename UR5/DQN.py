import gym
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyper parameters that will be used in the DQN algorithm
LEARNING_RATE = 0.00025
MEM_SIZE = 50000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.1
EPS_END = 0.0001
EPS_DECAY = 4 * MEM_SIZE
MEM_RETAIN = 0.1
NETWORK_UPDATE_ITERS = 5000
FC1_DIMS = 128
FC2_DIMS = 128

import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=bool)

    def add(self, state, action, reward, state_, done):
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            mem_index = int(self.mem_count % ((1 - MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))
        self.states[mem_index] = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] = 1 - done
        self.mem_count += 1

    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)
        self.target_network = Network(env)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.learn_count = 0

    def choose_action(self, observation):
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0

        if random.random() < eps_threshold:
            action = random.randint(0, 8)  # Randomly choose from 9 actions
        else:
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            self.policy_network.eval()
            with torch.no_grad():
                q_values = self.policy_network(state)
            action = torch.argmax(q_values).item()
        return action


    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=bool)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        self.policy_network.train(True)
        q_values = self.policy_network(states)
        q_values = q_values[batch_indices, actions]

        self.target_network.eval()
        with torch.no_grad():
            q_values_next = self.target_network(states_)
        q_values_next_max = torch.max(q_values_next, dim=1)[0]
        q_target = rewards + GAMMA * q_values_next_max * dones

        loss = self.policy_network.loss(q_target, q_values)
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        if self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def returning_epsilon(self):
        return self.exploration_rate