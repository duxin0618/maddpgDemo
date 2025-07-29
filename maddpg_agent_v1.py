import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0")

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, action_dim, batch_size):
        self.capacity = capacity

        self.obs_cap = np.empty((self.capacity, obs_dim))
        self.next_obs_cap = np.empty((self.capacity, obs_dim))
        self.state_cap = np.empty((self.capacity, state_dim))
        self.next_state_cap = np.empty((self.capacity, state_dim))
        self.action_cap = np.empty((self.capacity, action_dim))
        self.reward_cap = np.empty((self.capacity, 1))
        self.done_cap = np.empty((self.capacity, 1), dtype=np.bool)

        self.batch_size = batch_size
        self.current = 0

    def add_memo(self, obs, next_obs, state, next_state, action, reward, done):
        self.obs_cap[self.current] = obs
        self.next_obs_cap[self.current] = next_obs
        self.state_cap[self.current] = state
        self.next_state_cap[self.current] = next_state
        self.action_cap[self.current] = action
        self.reward_cap[self.current] = reward
        self.done_cap[self.current] = done

        self.current = (self.current + 1) % self.capacity

    def sample(self, idxes):
        obs = self.obs_cap[idxes]
        next_obs = self.next_obs_cap[idxes]
        state = self.state_cap[idxes]
        next_state = self.next_state_cap[idxes]
        action = self.action_cap[idxes]
        reward = self.reward_cap[idxes]
        done = self.done_cap[idxes]

        return obs, next_obs, state, next_state, action, reward, done



class Critic(nn.Module):
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, n_agent, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_dims + n_agent * action_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

    
class Actor(nn.Module):
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims, action_dims):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, action_dims)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.sigmoid(self.pi(x))
        # mu = torch.softmax(self.pi(x), dim=1)
        return mu
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))



class Agent:
    def __init__(self, memo_size, obs_dim, state_dim, n_agent, action_dim, lr_actor, lr_critic, fc1_dims, fc2_dims, gamma, tau, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(lr_actor=lr_actor, input_dims=obs_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, action_dims=action_dim).to(device)

        self.critic = Critic(lr_critic=lr_critic, input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim).to(device)

        self.target_actor = Actor(lr_actor=lr_actor, input_dims=obs_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, action_dims=action_dim).to(device)

        self.target_critic = Critic(lr_critic=lr_critic, input_dims=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim).to(device)

        self.replay_buffer = ReplayBuffer(capacity=memo_size, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim, batch_size=batch_size)

    
    def get_action(self, obs):
        single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0).to(device)
        single_action = self.actor.forward(single_obs)
        noise = torch.randn(self.action_dim).to(device) * 0.2
        single_action = torch.clamp(single_action+noise, min=0, max=1)

        return single_action.detach().cpu().numpy()[0]
    
    def save_model(self, filename):
        self.actor.save_checkpoint(filename)
        self.target_actor.save_checkpoint(filename)
        self.critic.save_checkpoint(filename)
        self.target_critic.save_checkpoint(filename)

        
