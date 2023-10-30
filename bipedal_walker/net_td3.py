import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle # for saving replaymemory
from net_sac import ReplayBuffer,CriticNetwork
from net_ddpg import ActorNetwork

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and T.cuda.is_available():
  mydevice=T.device('cuda')
else:
  mydevice=T.device('cpu')


class AgentTD3():
    def __init__(self, gamma, lr_a, lr_c, input_dims, batch_size, n_actions, n_hidden,
            action_space=None, max_mem_size=100, tau=0.001, name_prefix='', update_actor_interval=2, noise=0.1, hint_threshold=0.5, admm_rho=0.001, use_hint=False):
        self.gamma = gamma
        self.tau=tau
        self.batch_size = batch_size
        self.n_actions=n_actions
        # actions are always in [-1,1]
        self.max_action=1
        self.min_action=-1
        self.learn_step_cntr=0
        self.time_step=0
        self.update_actor_interval=update_actor_interval

        self.use_hint=use_hint
        if use_hint:
          self.admm_rho=admm_rho
          self.hint_threshold=hint_threshold
          self.rho=T.tensor(0.0,requires_grad=False,device=mydevice)
          self.zero_tensor=T.tensor(0.).to(mydevice)

        self.replaymem=ReplayBuffer(max_mem_size, input_dims, n_actions) 
    
        # online nets
        self.actor=ActorNetwork(lr_a, n_inputs=input_dims, n_actions=n_actions,
                n_hidden=n_hidden, name=name_prefix+'a_eval')
        self.critic_1=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_eval_1')
        self.critic_2=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_eval_2')
        # target nets
        self.target_actor=ActorNetwork(lr_a, n_inputs=input_dims, n_actions=n_actions, 
                n_hidden=n_hidden, name=name_prefix+'a_target')
        self.target_critic_1=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_target_1')
        self.target_critic_2=CriticNetwork(lr_c, n_inputs=input_dims, n_actions=n_actions, n_hidden=n_hidden, name=name_prefix+'q_target_2')
        # noise fraction
        self.noise = noise

        # initialize targets (hard copy)
        self.update_network_parameters(tau=1.)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        actor_state_dict = dict(actor_params)
        target_actor_params = self.target_actor.named_parameters()
        target_actor_dict = dict(target_actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)


        critic_params = self.critic_1.named_parameters()
        target_critic_params = self.target_critic_1.named_parameters()
        critic_state_dict = dict(critic_params)
        target_critic_dict = dict(target_critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()
        self.target_critic_1.load_state_dict(critic_state_dict)

        critic_params = self.critic_2.named_parameters()
        target_critic_params = self.target_critic_2.named_parameters()
        critic_state_dict = dict(critic_params)
        target_critic_dict = dict(target_critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()
        self.target_critic_2.load_state_dict(critic_state_dict)


    def store_transition(self, state, action, reward, state_, terminal, hint):
        self.replaymem.store_transition(state,action,reward,state_,terminal,hint)

    def choose_action(self, observation):
        self.actor.eval() # to disable batchnorm
        state = T.FloatTensor(observation).to(mydevice).unsqueeze(0)
        mu = self.actor.forward(state).to(mydevice)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise,size=(self.n_actions,)),
                                 dtype=T.float).to(mydevice)
        mu_prime = T.clamp(mu_prime,self.min_action,self.max_action)

        self.time_step +=1
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
        hint_batch = T.tensor(hint).to(mydevice)

        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        target_actions = self.target_actor.forward(new_state_batch)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action,
                                self.max_action)

        q1_ = self.target_critic_1.forward(new_state_batch, target_actions)
        q2_ = self.target_critic_2.forward(new_state_batch, target_actions)
        q1_[terminal_batch] = 0.0
        q2_[terminal_batch] = 0.0
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)
        critic_value_ = T.min(q1_, q2_)
        target = reward_batch + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.train()
        self.critic_2.train()

        def closure():
          if T.is_grad_enabled():
            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
          q1 = self.critic_1.forward(state_batch, action_batch)
          q2 = self.critic_2.forward(state_batch, action_batch)
          q1_loss = F.mse_loss(target, q1)
          q2_loss = F.mse_loss(target, q2)
          critic_loss = q1_loss + q2_loss
          if critic_loss.requires_grad:
            critic_loss.backward(retain_graph=True)

          return critic_loss

        self.critic_1.optimizer.step(closure)
        self.critic_2.optimizer.step(closure)


        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval == 0:
          self.actor.train()

          if not self.use_hint:
            self.actor.optimizer.zero_grad()
            actor_q1_loss = self.critic_1.forward(state_batch, self.actor.forward(state_batch))
            actor_loss = -T.mean(actor_q1_loss)
            actor_loss.backward()
            self.actor.optimizer.step()
          else:
            self.actor.optimizer.zero_grad()
            actions=self.actor.forward(state_batch)
            gfun=(T.max(self.zero_tensor,((F.mse_loss(actions, hint_batch)-self.hint_threshold)).mean()).pow(2))
            actor_q1_loss = self.critic_1.forward(state_batch, actions)
            actor_loss=-T.mean(actor_q1_loss)+0.5*self.admm_rho*gfun*gfun+self.rho*gfun
            actor_loss.backward()
            self.actor.optimizer.step()
            if self.learn_step_cntr % (100*self.update_actor_interval) == 0:
              with T.no_grad():
                self.rho+=self.admm_rho*gfun

          self.update_network_parameters()


    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
        self.replaymem.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
        self.replaymem.load_checkpoint()
        self.actor.train()
        self.target_actor.eval()
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self.update_network_parameters(tau=1.)

    def load_models_for_eval(self):
        self.actor.load_checkpoint_for_eval()
        self.critic_1.load_checkpoint_for_eval()
        self.critic_2.load_checkpoint_for_eval()
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def print(self):
        print(self.actor)
        print(self.critic_1)

#a=AgentTD3(gamma=0.99, batch_size=32, n_actions=2, max_mem_size=1024, input_dims=11,
#        n_hidden=10, lr_a=0.001, lr_c=0.001)
