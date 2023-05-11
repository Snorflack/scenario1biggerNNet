
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.utils import get_schedule_fn

from stable_baselines3.dqn.policies import CnnPolicy

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import gym
import torch as th
from torch import nn as nn
import gym_interface

from gym import spaces
import numpy as np
import random

import hexagdly

from collections import OrderedDict

class HexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hexConv2d = hexagdly.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        residual = x
        x = self.hexConv2d(x)
        if self.residual:
            x += residual
        x = self.relu(x)
        return x

class MyCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, n_residual_layers = 7, use_residual = False):
        super(MyCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        convs_per_layer = 64
        self.layers = OrderedDict()
        self.layers.update( {'conv': HexBlock(n_input_channels, convs_per_layer, residual=False)} )
        for i in range(n_residual_layers):
            layer_name = "resid"+str(i+1)
            self.layers.update( {layer_name: HexBlock(convs_per_layer, convs_per_layer, residual=use_residual)} ) 
        self.layers.update( {'flatten': nn.Flatten()})
        self.cnn = nn.Sequential(self.layers)
        model = self.cnn
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameter count {pytorch_total_params}")
        self.print_toggle = False

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.print_toggle:
            #print(observations)
            self.print_toggle = False
        return self.linear(self.cnn(observations))


# Comment out to use an automatically selected seed
# Uncomment for reproducible runs
# seed = 12345
# random.seed(seed)
# np.random.seed(seed)
# th.manual_seed(seed)

env = gym_interface.GymEnvironment(role="blue", versusAI="pass-agg", scenario="atomic-city-5x5.scn", saveReplay=False, actions19=False, ai="gym16", verbose=False, scenarioSeed=4025, scenarioCycle=1)

policy_kwargs = { "features_extractor_class" : MyCNN }
#policy = ActorCriticCnnPolicy # for PPO
policy = CnnPolicy # for DQN

#model = PPO(policy, env, clip_range=0.2, policy_kwargs=policy_kwargs, verbose=1)
model = DQN(policy, env, policy_kwargs=policy_kwargs, verbose=1)

# If doing additional training on an existing model, load here
#   using the appropriate model type and file name
#model = PPO.load("ppo_save.zip")  

model.set_env(env)
# n_eval_episodes should be as large as you can stand for scenarioCycle of 0, and at least the cycle length otherwise
model.learn(total_timesteps=1000, log_interval=10000, eval_env=env, eval_freq=300, n_eval_episodes=3, eval_log_path="lab3_logs") 

model.save("model_save")

# "deterministic" means using the maximum probability action always, as opposed to sampling from the distribution
print(f'eval results: {evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=False)}')

