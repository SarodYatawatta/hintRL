import argparse
import pickle
import time

import os, random
import numpy as np

import torch
import gym
#from net_ddpg import AgentDDPG
from net_td3 import AgentTD3

gamma=0.99
batch_size=256
lr=3e-4
hidden_size=256
tau=0.005
alpha=0.036
start_steps=10000
reward_scale = 5

parser = argparse.ArgumentParser()

def init_parser():
    parser.add_argument("--env_name", default="BipedalWalker-v3", help='environment name')  # OpenAI gym environment name
    parser.add_argument('--capacity', default=2097152, type=int, help='replaymemory size') # replay buffer size
    parser.add_argument('--iteration', default=100000, type=int, help='max episodes') #  num of  games
    parser.add_argument('--batch_size', default=256, type=int, help='batch size') # mini batch size
    parser.add_argument('--seed', default=10, type=int, help='random seed')
    parser.add_argument('--load', default=False, type=bool, help='load model')
    parser.add_argument('--use_hint', action='store_true', default=False, help='use hint')

init_parser()
args = parser.parse_args()

env = gym.make(args.env_name)

# Set seeds
env.reset(seed=args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

def main():
    agent = AgentTD3(gamma, lr, lr, state_dim, batch_size, action_dim, hidden_size, env.action_space, max_mem_size=args.capacity, tau=tau, name_prefix='learner', use_hint=args.use_hint, admm_rho=0.001, hint_threshold=0.5)
    if args.use_hint:
      hinter = AgentTD3(gamma, lr, lr, state_dim, batch_size, action_dim, hidden_size, env.action_space, max_mem_size=args.capacity, tau=tau, name_prefix='hinter', use_hint=False)
      hinter.load_models_for_eval()

    if args.load:
      agent.load_models()
    
    total_steps = 0
    scores=[]

    for i in range(args.iteration):
        ep_r = 0
        ep_s = 0
        done = False
        state, info = env.reset()

        while not done:
            action = []
            if total_steps < start_steps and not args.load:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            scaled_reward = reward * reward_scale if reward>0 else reward
            scores.append(reward)
            if not args.use_hint:
              agent.store_transition(state, action, scaled_reward, next_state, done, np.zeros(action_dim))
            else:
              hint  = hinter.choose_action(state)
              agent.store_transition(state, action, scaled_reward, next_state, done, hint)

            ep_r += reward
            ep_s += 1
            total_steps += 1

            state = next_state

        for ep in range(ep_s):
            agent.learn()

        if i>0 and (i==args.iteration-1 or i%500==0):
          agent.save_models()

        print(f'Ep {i}, steps {ep_s} of {total_steps}, score {ep_r}')

    
    env.close()


if __name__ == '__main__':
    
    main()
