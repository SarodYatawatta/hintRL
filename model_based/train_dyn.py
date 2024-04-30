import argparse
import pickle
import time

import os, sys, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lbfgsnew import LBFGSNew # custom optimizer
import gymnasium as gym

from models import EnsembleNet,ReplayBuffer
# Train a dynamics model (state,action) -> next state using a probabilistic network

use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

parser = argparse.ArgumentParser(description='Implementation of the PETS algorithm (Probabilistic ensembles with trajectory sampling)')

def init_parser():
    parser.add_argument("--env_name", default="BipedalWalker-v3", help='environment name')  # OpenAI gym environment name
    parser.add_argument("--optimizer", default="ADAM", help='Optimizer to use (ADAM or LBFGS)')  # Optimizer to use
    parser.add_argument('--iteration', default=30000, type=int, help='max episodes') #  num of  games
    parser.add_argument('--cycles', default=100, type=int, help='max train/predict cycles') #  num of  games
    parser.add_argument('--batch_size', default=256, type=int, help='batch size') # mini batch size
    parser.add_argument('--seed', default=10, type=int, help='random seed')

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

# Replay buffer
R=ReplayBuffer(120000,state_dim,action_dim)

#### using online estimation, find mean/var of output (state)
#### so the dynamics model can predict N(0,1) with normalization, this is scaled back to match the state distribution
niter=1
y_mean=np.zeros(state_dim+1)
y_moment=np.zeros(state_dim+1)
for i in range(30):
    done=False
    state,info=env.reset()
    while not done:
        action=env.action_space.sample()
        next_state, reward, terminated, truncated, info=env.step(action)
        done = terminated or truncated
        R.store_transition(state,action,reward,next_state,done)
        if not done:
         y=np.concatenate((state,np.expand_dims(reward,axis=0)/100.0))
         newmean=y_mean+(y-y_mean)/niter
         y_moment=y_moment+(y-y_mean)*(y-newmean)
         y_mean=newmean

         state=next_state
        
        niter+=1
    # check if replay buffer is full and break loop
    if R.is_full():
        break

print(f'Replay buffer filled: {R.mem_cntr} full capacity {R.is_full()}')

# add epsilon to avoid zero variance
y_std=np.sqrt(y_moment/(niter-1))+1e-6
y_mean=torch.FloatTensor(y_mean).to(mydevice)
y_std=torch.FloatTensor(y_std).to(mydevice)
# Override this: always set mean to zero and std to one
y_mean=torch.zeros_like(y_mean).to(mydevice)
y_std=torch.ones_like(y_std).to(mydevice)

# epochs for each learning step
n_epochs=10
# number of models in the ensemble
# input: state+action, output: next_state+reward
n_ensemble=5
n_candidates=25 # 500 candidate actions generated
n_particles=10 # 20 states sampled from the ensemble, model randomly selected from 1..n_ensemble
n_elites=5 # 8
cem_iter=5 # total iter=n_candidates x cem_iter
n_horizon=5 # 15 planning horizon length a_t,a_{t+1},...a_{t+T}: n_horizon+1 actions
n_hidden=4*64
model=EnsembleNet(n_ensemble,state_dim+action_dim,n_hidden,state_dim+1,out_mean=y_mean,out_std=y_std).to(mydevice)

if args.optimizer=='LBFGS' or args.optimizer=='Lbfgs':
  optimizers=[LBFGSNew(net.parameters(),history_size=7,line_search_fn=True,batch_mode=True) for net in model.net_list]
else:
  optimizers=[torch.optim.Adam(net.parameters(),lr=0.001) for net in model.net_list]

loss_func=nn.MSELoss()

#model.load_checkpoint()
#R.load_checkpoint()
######################################################################
# Train ensemble by bootstrapping replay buffer
def train_ensemble():
    total_steps = 0

    # intialize model
    #model.initialize()
    for i in range(args.iteration):
      for n in range(n_ensemble):
        state, action, reward, next_state, done = \
               R.sample_buffer_with_replacement(args.batch_size)
        x=torch.cat([torch.FloatTensor(state),torch.FloatTensor(action)],dim=1).to(mydevice)
        y=torch.cat([torch.FloatTensor(next_state),torch.FloatTensor(reward[:,None]/100.0)],dim=1).to(mydevice)

        def closure():
          if torch.is_grad_enabled():
            optimizers[n].zero_grad()
          # Minimizing the negative log likelihood
          y_dist=model.net_list[n].forward_dist(x)
          neg_log_likelihood=-y_dist.log_prob(y)
          loss=torch.mean(neg_log_likelihood)
          if loss.requires_grad:
            loss.backward()
            nn.utils.clip_grad_norm_(model.net_list[n].parameters(),max_norm=20,norm_type=2)
          return loss
        optimizers[n].step(closure)


        if total_steps % 10000 == 0:
           with torch.no_grad():
             yhat,_=model.net_list[n].forward(x)
             loss=loss_func(y,yhat)
             print(f'{n} {loss.data.item()}')
      total_steps+=1


# nll_beta: control weighting of data points (instead of default 1/var weight)
# nll_beta ~0 : high weight on small var, nll_beta~1 : equal weight : MSE
nll_beta=0.5
def train_ensemble1():
    total_steps = 0

    # intialize model
    #model.initialize()
    for i in range(args.iteration):
      for n in range(n_ensemble):
        state, action, reward, next_state, done = \
               R.sample_buffer_with_replacement(args.batch_size)
        x=torch.cat([torch.FloatTensor(state),torch.FloatTensor(action)],dim=1).to(mydevice)
        y=torch.cat([torch.FloatTensor(next_state),torch.FloatTensor(reward[:,None]/100.0)],dim=1).to(mydevice)

        # beta-NLL loss
        def closure():
          if torch.is_grad_enabled():
            optimizers[n].zero_grad()
          # Minimizing the negative log likelihood
          y_mean,y_var=model.net_list[n].forward(x)
          neg_log_likelihood=0.5*((y-y_mean)**2.0 / y_var +y_var.log() )

          if nll_beta>0:
           loss=neg_log_likelihood * (y_var.detach() ** nll_beta) 
          else:
           loss=neg_log_likelihood

          loss=torch.mean(loss)
          if loss.requires_grad:
            loss.backward()
            nn.utils.clip_grad_norm_(model.net_list[n].parameters(),max_norm=20,norm_type=2)
          return loss
        optimizers[n].step(closure)


        if total_steps % 10000 == 0:
           with torch.no_grad():
             yhat,_=model.net_list[n].forward(x)
             loss=loss_func(y,yhat)
             print(f'{n} {loss.data.item()}')
      total_steps+=1




######################################################################
def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0, std: float = 1,
    lower_bound: torch.Tensor=-1, upper_bound: torch.Tensor=1
) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < lower_bound, tensor > upper_bound)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(
            mean, std, size=(bound_violations,), device=tensor.device
        )
    return tensor

def CEM_optimize(state0, bound_low, bound_high):
    # initial mean, std
    mu=torch.zeros(action_dim)*0.5
    sigma=torch.ones(action_dim)*1.0

    state=torch.FloatTensor(state0).to(mydevice)
    for i in range(cem_iter):
        # storage for candidate a_t, and average reward
        cand_actions=torch.zeros(n_candidates,action_dim).to(mydevice)
        avg_reward=torch.zeros(n_candidates).to(mydevice)
        for cand in range(n_candidates):
          # sample to generate a_t,a_{t+1}, use tanh() to limit to [-1,1]
          a_t=torch.tanh(torch.normal(mu,sigma))
          a_t=a_t.to(mydevice)
          cand_actions[cand]=a_t
          # generate n_horizon actions (trajectory length=n_horizon+1)
          a_t_1=torch.zeros(n_horizon,action_dim).to(mydevice)
          for h in range(n_horizon):
            a_t_1[h]=torch.tanh(torch.normal(mu,sigma))
          a_t_1=a_t_1.to(mydevice)
          x_t=torch.cat([state,a_t]).to(mydevice)
          # generate particles
          for n in range(n_particles):
              # select model
              b=np.random.choice(np.arange(n_ensemble))
              # generate state
              y=model.net_list[b].sample(x_t)
              s_t_1=y[:-1]
              reward_t=y[-1]*100.0
              avg_reward_traject=reward_t
              for h in range(n_horizon):
                x_t_1=torch.cat([s_t_1,a_t_1[h]]).to(mydevice)
                y=model.net_list[b].sample(x_t_1)
                s_t_1=y[:-1]
                reward_t_1=y[-1]*100.0
                # average reward over trajectory
                avg_reward_traject+=reward_t_1
              # add this to candidate action reward
              avg_reward[cand]=avg_reward[cand]+avg_reward_traject/(n_horizon+1.0)
          # average reward over particles
          avg_reward[cand]=avg_reward[cand]/n_particles
        v=torch.topk(avg_reward,n_elites)
        top_actions=cand_actions[v.indices]
        mu=torch.mean(top_actions,dim=0)
        sigma=torch.sqrt(torch.var(top_actions,dim=0))+1e-3

    return mu.cpu().numpy()
######################################################################
# Trajectory sampling
def learn():
    with torch.no_grad():
      for i in range(n_epochs):
        ep_r=0
        ep_s=0
        state,_ = env.reset()
        done=False
        while not done:
           #  perform trajectory sampling, starting from current state to get the action
           action = CEM_optimize(torch.FloatTensor(state),env.action_space.low, env.action_space.high)
           #action = env.action_space.sample()
           next_state, reward, terminated, truncated, info=env.step(action)
           done = terminated or truncated
           R.store_transition(state,action,reward,next_state,done)
           ep_r += reward
           ep_s += 1
           #print(f'{i} {ep_s} {ep_r}')

        print(f'Ep {i} steps {ep_s} score {ep_r}')

######################################################################
def main():
    for i in range(args.cycles):
      print(f'starting major loop {i}')
      train_ensemble1()
      model.save_checkpoint()
      R.save_checkpoint()
      learn()
      sys.stdout.flush()

if __name__ == '__main__':
    main()

# python train_dyn.py --iteration 400 --cycles 100 --env_name BipedalWalkerHardcore-v3 --optimizer LBFGS
