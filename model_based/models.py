import os,argparse
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

# initialize all layer weights, based on the fan in
def init_layer(layer,sc=None):
  sc = sc or 1./np.sqrt(layer.weight.data.size()[0])
  torch.nn.init.uniform_(layer.weight.data, -sc, sc)
  torch.nn.init.uniform_(layer.bias.data, -sc, sc)

# dynamics model (state,action) -> next state using a probabilistic network
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output,name,out_mean=0,out_std=1):
        super().__init__()
        self.shared=nn.Linear(n_input,n_hidden)
        self.mean_l1=nn.Linear(n_hidden,n_hidden)
        self.mean_l2=nn.Linear(n_hidden,n_hidden)
        self.mean_l3=nn.Linear(n_hidden,n_output)
        self.logstd_l1=nn.Linear(n_hidden,n_hidden)
        self.logstd_l2=nn.Linear(n_hidden,n_hidden)
        self.logstd_l3=nn.Linear(n_hidden,n_output)
        # if given, use mu,sigma to scale from N(0,1) to N(mu,sigma I)
        self.y_mu=out_mean
        self.y_sigma=out_std

        init_layer(self.shared)
        init_layer(self.mean_l1)
        init_layer(self.mean_l2)
        init_layer(self.mean_l3)
        init_layer(self.logstd_l1)
        init_layer(self.logstd_l2)
        init_layer(self.logstd_l3)

        self.max_logstd=0.5
        self.min_logstd=-10.0
        self.checkpoint_file = os.path.join('./', name+'_prob.model')

        self.to(mydevice)

    def forward(self,x):
        # TBD normalize data 'x' 
        x=F.elu(self.shared(x))
        xmean=F.elu(self.mean_l1(x))
        xmean=F.elu(self.mean_l2(xmean))
        xmean=self.mean_l3(xmean)
        xlogvar=F.elu(self.logstd_l1(x))
        xlogvar=F.elu(self.logstd_l2(xlogvar))
        xlogvar=self.logstd_l3(xlogvar)
        # restrict logstd to the range
        xlogvar = self.max_logstd - F.softplus(self.max_logstd-xlogvar)
        xlogvar = self.min_logstd + F.softplus(xlogvar - self.min_logstd)
        xstd=xlogvar.exp()

        # scale from N(0,1) to N(mu,sigma)
        xmean=xmean+self.y_mu
        xstd=xstd*self.y_sigma

        return xmean,xstd

    def rsample(self,x):
        xmean,xstd=self.forward(x)
        problayer=torch.distributions.Normal(xmean,xstd)
        x=problayer.rsample()

        return x

    def sample(self,x):
        with torch.no_grad():
          xmean,xstd=self.forward(x)
          problayer=torch.distributions.Normal(xmean,xstd)
          x=problayer.sample()

          return x

    def forward_dist(self,x):
        # instead of returning a value, we return the distribution object
        xmean,xstd=self.forward(x)

        problayer=torch.distributions.Normal(xmean,xstd)
        return problayer

    def initialize(self):
        init_layer(self.shared)
        init_layer(self.mean_l1)
        init_layer(self.mean_l2)
        init_layer(self.logstd_l1)
        init_layer(self.logstd_l2)
        
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class EnsembleNet(nn.Module):
    def __init__(self,n_ensemble,n_input,n_hidden,n_output,out_mean=0,out_std=1):
        super(EnsembleNet,self).__init__()
        self.net_list=nn.ModuleList([Net(n_input,n_hidden,n_output,str(k),out_mean,out_std) for k in range(n_ensemble)])

    def forward(self,x,k=None):
        if k is not None:
            return self.net_list[k](x)
        else:
            return [net(x) for net in self.net_list]

    def initialize(self):
        for net in self.net_list:
            net.initialize()

    def save_checkpoint(self):
        for net in self.net_list:
            net.save_checkpoint()

    def load_checkpoint(self):
        for net in self.net_list:
            net.load_checkpoint()
            net.to(mydevice)

class ReplayBuffer(object):
    def __init__(self,max_size,input_shape,n_actions,name_prefix=''):
        self.mem_size=max_size
        self.mem_cntr=0
        self.state_memory=np.zeros((self.mem_size,input_shape),dtype=np.float32)
        self.new_state_memory=np.zeros((self.mem_size,input_shape),dtype=np.float32)
        self.action_memory=np.zeros((self.mem_size,n_actions),dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.filename=name_prefix+'replaymem_prob.model' # for saving object


    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.state_memory[index] = state
        self.new_state_memory[index] = state_

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def sample_buffer_with_replacement(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=True)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
          pickle.dump(self,f)

    def is_full(self):
        return self.mem_cntr > self.mem_size

    def load_checkpoint(self):
        with open(self.filename,'rb') as f:
          temp=pickle.load(f)
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          self.state_memory=temp.state_memory
          self.new_state_memory=temp.new_state_memory
          self.action_memory=temp.action_memory
          self.reward_memory=temp.reward_memory
          self.terminal_memory=temp.terminal_memory

#model=EnsembleNet(3,100,64,20,out_mean=0.0,out_std=1.0)
#print(model)
#print(model.net_list[0])
#R=ReplayBuffer(1024,100,32,'test')
#print(R)
