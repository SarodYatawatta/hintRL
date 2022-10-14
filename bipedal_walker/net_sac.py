import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import pickle # for saving replaymemory
from PER import PER

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

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, name_prefix=''):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.hint_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.filename=name_prefix+'replaymem_sac.model' # for saving object

    def store_transition(self, state, action, reward, state_, done, hint):
        index = self.mem_cntr % self.mem_size
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.hint_memory[index] = hint

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        hint = self.hint_memory[batch]

        return states, actions, rewards, states_, terminal, hint

    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
          pickle.dump(self,f)
        
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
          self.hint_memory=temp.hint_memory
 
# input: state,action output: q-value
class CriticNetwork(nn.Module):
    def __init__(self, lr, n_inputs, n_actions, n_hidden, name):
        super(CriticNetwork, self).__init__()
        self.n_inputs = n_inputs # state dims
        self.n_actions = n_actions # action dims
        self.fc1 = nn.Linear(n_inputs + n_actions, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3,0.003)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_sac_critic.model')

        self.to(self.device)

    def forward(self, state, action):
        x0 = T.cat([state, action], 1)
        x = F.relu(self.fc1(x0))
        x = F.relu(self.fc2(x))
        x1 = self.fc3(x)

        return x1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))

# input: state output: action
class ActorNetwork(nn.Module):
    def __init__(self, lr, n_inputs, n_actions, n_hidden, name, action_space=None):
        super(ActorNetwork, self).__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3mu = nn.Linear(n_hidden, n_actions)
        self.fc3logsigma = nn.Linear(n_hidden, n_actions)

        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3mu,0.003)
        init_layer(self.fc3logsigma,0.003) # last layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = mydevice
        self.checkpoint_file = os.path.join('./', name+'_sac_actor.model')

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

    def forward(self, state):
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        mu=self.fc3mu(x) # 
        logsigma=self.fc3logsigma(x) # 
        logsigma = T.clamp(logsigma, min=-20, max=2)

        return mu,logsigma 

    def sample_normal(self, state, reparameterize=True):
        mu, logsigma = self.forward(state)
        sigma=logsigma.exp()
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        # add batch dimension if missing
        if actions.dim()==1:
         actions.unsqueeze_(0)

        actions_t=T.tanh(actions)
        # scale actions
        action = (actions_t * self.action_scale + self.action_bias).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(self.action_scale*(1-actions_t.pow(2))+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_checkpoint_for_eval(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))


class Agent():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions, n_hidden,
            action_space, alpha, max_mem_size=100, tau=0.001, name_prefix='', hint_threshold=0.5, admm_rho=0.001, use_hint=False, prioritized=False):
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.n_actions=n_actions
        
        self.prioritized=prioritized
        if not self.prioritized:
          self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions, name_prefix) 
        else:
          self.replaymem=PER(max_mem_size, input_dims, n_actions, name_prefix)

        # online nets
        self.actor=ActorNetwork(lr_a, n_inputs=input_dims, n_actions=n_actions,
                n_hidden=n_hidden, name=name_prefix+'a_eval')
        self.critic_1=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_eval_1')
        self.critic_2=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_eval_2')
        # target nets
        self.target_critic_1=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_target_1')
        self.target_critic_2=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_target_2')

        # temperature
        self.alpha=T.tensor(alpha,requires_grad=False,device=mydevice)
        self.zero_tensor=T.tensor(0.).to(mydevice)

        self.learn_alpha=False
        if self.learn_alpha:
           # target entropy: -number of bits to represent the state
           self.target_entropy = -np.sum(action_space.shape)
           self.alpha_lr = 1e-4

        self.use_hint=use_hint
        if use_hint:
          self.hint_threshold=hint_threshold
          self.rho=T.tensor(0.0,requires_grad=False,device=mydevice)
          self.admm_rho=admm_rho

        # initialize targets (hard copy)
        self.update_network_parameters(tau=1.)

        self.learn_counter=0

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        v_params = self.critic_1.named_parameters()
        v_dict = dict(v_params)
        target_v_params = self.target_critic_1.named_parameters()
        target_v_dict = dict(target_v_params)
        for name in target_v_dict:
            target_v_dict[name] = tau*v_dict[name].clone() + \
                                      (1-tau)*target_v_dict[name].clone()
        self.target_critic_1.load_state_dict(target_v_dict)

        v_params = self.critic_2.named_parameters()
        v_dict = dict(v_params)
        target_v_params = self.target_critic_2.named_parameters()
        target_v_dict = dict(target_v_params)
        for name in target_v_dict:
            target_v_dict[name] = tau*v_dict[name].clone() + \
                                      (1-tau)*target_v_dict[name].clone()
        self.target_critic_2.load_state_dict(target_v_dict)

    def store_transition(self, state, action, reward, state_, terminal, hint):
        self.replaymem.store_transition(state,action,reward,state_,terminal, hint)

    def choose_action(self, observation):
        state = T.FloatTensor(observation).to(mydevice).unsqueeze(0)
        self.actor.eval() # to disable batchnorm
        actions,_= self.actor.sample_normal(state, reparameterize=False)
        self.actor.train() # to enable batchnorm

        return actions.cpu().detach().numpy()[0]

    def learn(self):
        if self.replaymem.mem_cntr < self.batch_size:
            return

        if not self.prioritized:
           state, action, reward, new_state, done, hint = \
                                self.replaymem.sample_buffer(self.batch_size)
        else:
           state, action, reward, new_state, done, hint, idxs, is_weights = \
                                self.replaymem.sample_buffer(self.batch_size)

        state_batch = T.tensor(state).to(mydevice)
        new_state_batch = T.tensor(new_state).to(mydevice)
        action_batch = T.tensor(action).to(mydevice)
        reward_batch = T.tensor(reward).to(mydevice).unsqueeze(1)
        terminal_batch = T.tensor(done).to(mydevice).unsqueeze(1)
        hint_batch = T.tensor(hint).to(mydevice)

        if self.prioritized:
            is_weight=T.tensor(is_weights).to(mydevice)
        
        with T.no_grad():
            new_actions, new_log_probs = self.actor.sample_normal(new_state_batch, reparameterize=False)
            q1_new_policy = self.target_critic_1.forward(new_state_batch, new_actions)
            q2_new_policy = self.target_critic_2.forward(new_state_batch, new_actions)
            min_next_target=T.min(q1_new_policy,q2_new_policy)-self.alpha*new_log_probs
            min_next_target[terminal_batch]=0.0
            new_q_value=reward_batch+self.gamma*min_next_target

        q1_new_policy = self.critic_1.forward(state_batch, action_batch)
        q2_new_policy = self.critic_2.forward(state_batch, action_batch)

        if self.prioritized:
            errors1=T.abs(q1_new_policy-new_q_value).detach().cpu().numpy()
            errors2=T.abs(q2_new_policy-new_q_value).detach().cpu().numpy()
            self.replaymem.batch_update(idxs,0.5*(errors1+errors2))

        if not self.prioritized:
            critic_1_loss = F.mse_loss(q1_new_policy, new_q_value)
            critic_2_loss = F.mse_loss(q2_new_policy, new_q_value)
        else:
            critic_1_loss = self.replaymem.mse(q1_new_policy, new_q_value, is_weight)
            critic_2_loss = self.replaymem.mse(q2_new_policy, new_q_value, is_weight)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state_batch, reparameterize=True)
        q1_new_policy = self.critic_1.forward(state_batch, actions)
        q2_new_policy = self.critic_2.forward(state_batch, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        if not self.use_hint:
          if not self.prioritized:
            actor_loss = (self.alpha*log_probs - critic_value).mean()
          else:
            actor_loss = (is_weight*(self.alpha*log_probs - critic_value)).mean()
          self.actor.optimizer.zero_grad()
          actor_loss.backward()
          self.actor.optimizer.step()
        else:
             if self.prioritized:
               gfun=(T.max(self.zero_tensor,(is_weight*(F.mse_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))
             else:
               gfun=(T.max(self.zero_tensor,((F.mse_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))

             if not self.prioritized:
                actor_loss = (self.alpha*log_probs - critic_value).mean()+0.5*self.admm_rho*gfun*gfun+self.rho*gfun
             else:
                actor_loss = (is_weight*(self.alpha*log_probs - critic_value)).mean()+0.5*self.admm_rho*gfun*gfun+self.rho*gfun
             self.actor.optimizer.zero_grad()
             actor_loss.backward()
             self.actor.optimizer.step()


        if self.learn_counter%10==0:
          if self.learn_alpha or self.use_hint:
            with T.no_grad():
               actions, log_probs = self.actor.sample_normal(state_batch, reparameterize=False)
               if self.learn_alpha:
                   if self.prioritized:
                     self.alpha=T.max(self.zero_tensor,self.alpha+self.alpha_lr*(is_weight*(self.target_entropy-(-log_probs))).mean())
                   else:
                     self.alpha=T.max(self.zero_tensor,self.alpha+self.alpha_lr*((self.target_entropy-(-log_probs)).mean()))

               if self.use_hint:
                   if self.prioritized:
                      gfun=(T.max(self.zero_tensor,(is_weight*(F.mse_loss(actions, hint_batch)-self.hint_threshold).detach()).mean()).pow(2))
                   else:
                      gfun=(T.max(self.zero_tensor,((F.mse_loss(actions, hint_batch)-self.hint_threshold).detach()).mean()).pow(2))
                   self.rho+=self.admm_rho*gfun

        if self.learn_counter%10000==0:
          if self.use_hint and self.learn_alpha:
             print(f'{self.learn_counter} {self.rho} {self.alpha}')
          elif self.use_hint: 
             print(f'{self.learn_counter} {self.rho}')
          elif self.learn_alpha: 
             print(f'{self.learn_counter} {self.alpha}')

        self.learn_counter+=1
        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.replaymem.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.replaymem.load_checkpoint()
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.update_network_parameters(tau=1.)

    def load_models_for_eval(self):
        self.actor.load_checkpoint_for_eval()
        self.critic_1.load_checkpoint_for_eval()
        self.critic_2.load_checkpoint_for_eval()
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.update_network_parameters(tau=1.)

    def disable_hint(self):
        self.use_hint=False

    def print(self):
        print(self.actor)
        print(self.critic_1)

#a=Agent(gamma=0.99, batch_size=32, n_actions=2,  
#                max_mem_size=1000, input_dims=11, n_hidden=10, lr_a=0.001, lr_c=0.001)
