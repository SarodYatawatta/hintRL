import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle # for saving replaymemory

from net_sac import ReplayBuffer,CriticNetwork 

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and T.cuda.is_available():
  mydevice=T.device('cuda')
else:
  mydevice=T.device('cpu')

# initialize all layer weights, based on the fan in
def init_layer(layer,sc=None):
  sc = sc or 1./np.sqrt(layer.weight.data.size()[0])
  T.nn.init.uniform_(layer.weight.data, -sc, sc)
  T.nn.init.uniform_(layer.bias.data, -sc, sc)

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

# input: state output: action
class ActorNetwork(nn.Module):
    def __init__(self, lr, n_inputs, n_actions, n_hidden, name, action_space=None):
        super(ActorNetwork, self).__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3,0.003)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_ddpg_actor.model')
        # action rescaling
        if action_space is None:
            self.action_scale = T.tensor(1.)
            self.action_bias = T.tensor(0.)
        else:
            self.action_scale = T.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = T.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.to(self.device)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        actions=T.tanh(self.fc3(x)) # in [-1,1], shift, scale up as needed (in the environment)

        return actions*self.action_scale+self.action_bias

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))


class AgentDDPG():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions, n_hidden, action_space=None, max_mem_size=100, tau=0.001, name_prefix=''):
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size

        self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions, name_prefix) 
    
        # current net
        self.actor=ActorNetwork(lr_a, n_inputs=input_dims, n_actions=n_actions, 
                n_hidden=n_hidden, name=name_prefix+'a_eval')
        self.critic=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden,  name=name_prefix+'q_eval')
        # target net
        self.target_actor=ActorNetwork(lr_a, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden,
                name=name_prefix+'a_target')
        self.target_critic=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_target')
        # noise with memory
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # initialize targets (hard copy)
        self.update_network_parameters(tau=1.)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def store_transition(self, state, action, reward, state_, terminal, hint):
        self.replaymem.store_transition(state,action,reward,state_,terminal, hint)

    def choose_action(self, observation):
        state = T.FloatTensor(observation).to(mydevice).unsqueeze(0)
        self.actor.eval() # to disable batchnorm
        mu = self.actor.forward(state).to(mydevice)
        self.actor.train() # to enable batchnorm
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(mydevice)
        return mu_prime.cpu().detach().numpy()[0]

    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return

        
        state, action, reward, new_state, done, hint = \
                                self.replaymem.sample_buffer(self.batch_size)

        state_batch = T.tensor(state).to(mydevice)
        new_state_batch = T.tensor(new_state).to(mydevice)
        action_batch = T.tensor(action).to(mydevice)
        reward_batch = T.tensor(reward).to(mydevice)
        terminal_batch = T.tensor(done).to(mydevice)

        self.target_actor.eval()
        self.target_critic.eval()
        target_actions = self.target_actor.forward(new_state_batch)
        critic_value_ = self.target_critic.forward(new_state_batch, target_actions)

        next_target = critic_value_
        next_target[terminal_batch]=0.0
        target=reward_batch+self.gamma*next_target

        self.critic.train()
        def closure():
          if T.is_grad_enabled():
            self.critic.optimizer.zero_grad()
          critic_value = self.critic.forward(state_batch, action_batch)
          bellman_error=(critic_value-target)# dont clip .clamp(-1,1)
          critic_loss=T.norm(bellman_error,2)**2
          if critic_loss.requires_grad:
            critic_loss.backward(retain_graph=True)
          return critic_loss
        self.critic.optimizer.step(closure)
        self.critic.eval()

        self.actor.train()
        def closure1():
          if T.is_grad_enabled():
            self.actor.optimizer.zero_grad()
          mu = self.actor.forward(state_batch)
          actor_loss = -self.critic.forward(state_batch, mu)
          actor_loss = T.mean(actor_loss)
          if actor_loss.requires_grad:
            actor_loss.backward(retain_graph=True)
          return actor_loss
        self.actor.optimizer.step(closure1)
        self.actor.eval()

        self.update_network_parameters()


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        self.replaymem.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.replaymem.load_checkpoint()
        self.actor.train()
        self.target_actor.eval()
        self.critic.train()
        self.target_critic.eval()

    def load_models_for_eval(self):
        self.actor.load_checkpoint_for_eval()
        self.critic.load_checkpoint_for_eval()
        self.actor.eval()
        self.critic.eval()

    def print(self):
        print(self.actor)
        print(self.critic)

#a=AgentDDPG(gamma=0.99, batch_size=32, n_actions=2,
#    max_mem_size=1000, input_dims=11, n_hidden=10, lr_a=0.001, lr_c=0.001)
