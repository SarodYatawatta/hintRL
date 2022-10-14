import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle # for saving replaymemory
import os # for saving networks 

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and T.cuda.is_available():
  mydevice=T.device('cuda')
else:
  mydevice=T.device('cpu')

########################################
# Prioritized experience replay memory
# From https://github.com/Ullar-Kask/TD3-PER
def is_power_of_2 (n):
    return ((n & (n - 1)) == 0) and n != 0

# A binary tree data structure where the parent’s value is the sum of its children
class SumTree():
    #"""
    #This SumTree code is modified version of the code from:
    #https://github.com/jaara/AI-blog/blob/master/SumTree.py
    #https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
    #For explanations please see:
    #https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    #"""
    data_pointer = 0
    data_length = 0
    
    def __init__(self, capacity):
        # Initialize the tree with all nodes = 0,
        
        # Number of leaf nodes (final nodes) that contains experiences
        # Should be power of 2.
        self.capacity = int(capacity)
        assert is_power_of_2(self.capacity), "Capacity must be power of 2."
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        #""" tree:
        #    0
        #   / \
        #  0   0
        # / \ / \
        #0  0 0  0  [Size: capacity] it's at this line where the priority scores are stored
        #"""
        
        # Contains the experiences (so the size of data is capacity)
        #self.data = np.zeros(capacity, dtype=object)
    
    def __len__(self):
        return self.data_length
    
    def add(self, priority):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        # Index to update data frame
        data_index=self.data_pointer
        #self.data[self.data_pointer] = data
        # Update the leaf
        self.update (tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.data_length < self.capacity:
            self.data_length += 1
        return data_index
    
    def update(self, tree_index, priority):
        # change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:
            #"""
            #Here we want to access the line above
            #THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            # 
            #    0
            #   / \
            #  1   2
            # / \ / \
            #3  4 5  [6] 
            # 
            #If we are in leaf at index 6, we updated the priority score
            #We need then to update index 2 node
            #So tree_index = (tree_index - 1) // 2
            #tree_index = (6-1)//2
            #tree_index = 2 (because // round the result)
            #"""
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change    
    
    def get_leaf(self, v):
        # Get the leaf_index, priority value of that leaf and experience associated with that index
        #"""
        #Tree structure and array storage:
        #Tree index:
        #     0         -> storing priority sum
        #    / \
        #  1     2
        # / \   / \
        #3   4 5   6    -> storing priority for experiences
        #Array type for storing:
        #[0,1,2,3,4,5,6]
        #"""
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        
        return leaf_index, self.tree[leaf_index], data_index
    
    @property
    def total_priority(self):
        return self.tree[0]  # the root


class PER(object):  # stored as ( s, a, r, s_new, done ) in SumTree
    #"""
    #This PER code is modified version of the code from:
    #https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    #"""
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0..1] convert the importance of TD error to priority, often 0.6
    beta = 0.4  # importance-sampling, from initial value increasing to 1, often 0.4
    beta_increment_per_sampling = 1e-4  # annealing the bias, often 1e-3
    absolute_error_upper = 1.   # clipped abs error
    mem_cntr=0
    
    def __init__(self, capacity, input_shape, n_actions, name_prefix=''):
        #"""
        #The tree is composed of a sum tree that contains the priority scores at his leaf and also a indices to data arrays.
        #capacity: should be a power of 2
        #"""
        self.tree = SumTree(capacity)
        self.mem_size=capacity
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.hint_memory = np.zeros((self.mem_size,n_actions), dtype=np.float32)

        self.filename=name_prefix+'prioritized_replaymem_sac.model'
    
    def __len__(self):
        return len(self.tree)
    
    def is_full(self):
        return len(self.tree) >= self.tree.capacity
    
    def store_transition(self, state, action, reward, state_, done, hint, error = None):
        if error is None:
            priority = np.amax(self.tree.tree[-self.tree.capacity:])
            if priority == 0: priority = self.absolute_error_upper
        else:
            priority = min((abs(error) + self.epsilon) ** self.alpha, self.absolute_error_upper)
        index=self.tree.add(priority)
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.hint_memory[index]=hint

        self.mem_cntr+=1
    
    def sample_buffer(self, batch_size):
        #"""
        #- First, to sample a minibatch of size k the range [0, priority_total] is divided into k ranges.
        #- Then a value is uniformly sampled from each range.
        #- We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        #- Then, we calculate IS weights for each minibatch element.
        #"""
        
        minibatch = []
        
        idxs = np.empty((batch_size,), dtype=np.int32)
        is_weights = np.empty((batch_size,), dtype=np.float32)
        data_idxs = np.empty((batch_size,), dtype=np.int32)
        
        # Calculate the priority segment
        # Divide the Range[0, ptotal] into batch_size ranges
        priority_segment = self.tree.total_priority / batch_size # priority segment
        # Increase the beta each time we sample a new minibatch
        self.beta = np.amin([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        for i in range(batch_size):
            #"""
            #A value is uniformly sampled from each range
            #"""
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            #"""
            #Experience that corresponds to each value is retrieved
            #"""
            index, priority, data_index = self.tree.get_leaf(value)
            sampling_probabilities = priority / (self.tree.total_priority)
            is_weights[i] = np.power(batch_size * (sampling_probabilities+self.epsilon), -self.beta)
            idxs[i]= index
            data_idxs[i]=data_index

        is_weights /= is_weights.max()
            
        states=self.state_memory[data_idxs]
        actions=self.action_memory[data_idxs]
        rewards=self.reward_memory[data_idxs]
        states_=self.new_state_memory[data_idxs]
        terminal=self.terminal_memory[data_idxs]
        hints=self.hint_memory[data_idxs]
        return states, actions, rewards, states_, terminal, hints, idxs, is_weights
    
    def batch_update(self, idxs, errors):
        #"""
        #Update the priorities on the tree
        #"""
        errors = errors + self.epsilon
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)
        
        for idx, p in zip(idxs, ps):
            self.tree.update(idx, p)
    
    # custom mean squared error with importance sampling weights
    def mse(self,expected,targets,is_weights):
        td_error=expected-targets
        weighted_squared_error=is_weights*td_error*td_error
        return T.sum(weighted_squared_error)/T.numel(weighted_squared_error)

    def save_checkpoint(self):
        with open(self.filename,'wb') as f:
            pickle.dump(self,f)

    def load_checkpoint(self):
        with open(self.filename,'rb') as f:
            temp=pickle.load(f)
            self.tree=temp.tree
            self.mem_size=temp.mem_size
            self.mem_cntr=temp.mem_cntr
            self.state_memory=temp.state_memory
            self.new_state_memory=temp.new_state_memory
            self.action_memory=temp.action_memory
            self.reward_memory=temp.reward_memory
            self.terminal_memory=temp.terminal_memory
            self.hint_memory=temp.hint_memory

    # normalize rewards
    def normalize_reward(self):
          mu=self.reward_memory[:self.mem_cntr].mean()
          sigma=self.reward_memory[:self.mem_cntr].std()
          self.reward_memory[:self.mem_cntr]=(self.reward_memory[:self.mem_cntr]-mu)/sigma
