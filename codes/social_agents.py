import torch
import numpy as np
from torch.utils.data import Dataset
from itertools import count
from agent import Agent
from collections import deque
from networks import BCNetwork
import torch.optim as optim
from torch.distributions import Categorical
from imitation.util.util import make_vec_env
import gymnasium as gym
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from imitation.data.wrappers import RolloutInfoWrapper
from utils import ExpertDataset
from utils import evaluate_agent
from imitation.data import rollout



class OnlineSocialAgent(Agent):
    def __init__(self, args, social_agents_list):

        super().__init__(args)

        self.social_agents_list = social_agents_list
        self.BC_agents_list = []
        self.BC_train_steps = [10,20,50,100,200,350]
        self.BC_train_epochs = [500,500,200,100,100,100]
        self.rollouts = []
        self.current_agent = 0
        # self.avg_alpha = -0.1
        self.env_max_score = 200
        self.selected_agent_list = [0]
        self.n_agents = len(social_agents_list) + 1
        self.agents_normalized_scores = torch.zeros(self.n_agents)
        self.deque_len = 15
        self.scores_lists = [deque(maxlen=self.deque_len) for _ in range(self.n_agents)]
        
        BC_hardcore_expert = BCNetwork(self.state_size, self.action_size).to(self.args.device)
        self.BC_agents_list.append(BC_hardcore_expert)
        self.BC_agents_list = self.BC_agents_list + social_agents_list[1:]
        self.BC_optimizer = optim.Adam(BC_hardcore_expert.parameters(), lr=0.001)


    def update_score_values(self, new_score):
        self.scores_lists[self.current_agent].append(new_score / self.env_max_score * 2)
        self.agents_normalized_scores[self.current_agent] = np.mean(self.scores_lists[self.current_agent])
            # weights=np.exp(-self.avg_alpha * np.arange(len(self.scores_lists[self.current_agent]))))       
        # self.agents_normalized_scores[0] = max(0.0, self.agents_normalized_scores[0])
        dist = Categorical(logits=self.agents_normalized_scores)
        self.current_agent = int(dist.sample())
        self.selected_agent_list.append(self.current_agent)
        return

    """ behavior cloning is only used for hardcore agent.
        behavior cloning will learn the flat expert policy
        in just 10 epidodes an it also gains low rewards
    """
    def update_BC_agents(self, i_episode):
        
        if i_episode not in self.BC_train_steps:
            return
        else: #update only in selected episodes
            env = gym.make("BipedalWalker-v3", max_episode_steps=750, hardcore = True)
            
            venv = make_vec_env(
                "BipedalWalker-v3",
                rng=np.random.default_rng(),
                post_wrappers=[
                    lambda env, _: RolloutInfoWrapper(env)
                ],
                env_make_kwargs={"hardcore": True}
            )
            train_step_idx = self.BC_train_steps.index(i_episode)
            transitions = self.generate_transitions(venv, train_step_idx)
            
            
            dataset = ExpertDataset(transitions)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            
        self.train_BC_agent(dataloader, train_step_idx)
        score_mean, score_std = evaluate_agent(self.BC_agents_list[0])
        
        print(f'Hardcore agent updated at episode {i_episode} with mean {score_mean:.2f} and std {score_std:02f}')
            
            
    def generate_transitions(self, venv, train_step_idx):
            
        episodes_to_add = (
            self.BC_train_steps[train_step_idx] - self.BC_train_steps[train_step_idx - 1]
            if train_step_idx > 0
            else self.BC_train_steps[0]
        )        
        rollouts = rollout.rollout(
        self.social_agents_list[0],
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=episodes_to_add-7),
        rng=np.random.default_rng(),)

        self.rollouts = self.rollouts + rollouts
        transitions = rollout.flatten_trajectories(self.rollouts)
        
        return transitions
    
    def train_BC_agent(self, dataloader, train_step_idx):    
        criterion = nn.MSELoss()
        BC_agent = self.BC_agents_list[0]
        
        # Training loop
        num_epochs = self.BC_train_epochs[train_step_idx]
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_states, batch_actions in dataloader:
                
                # Forward pass
                predicted_actions = BC_agent(batch_states)
                
                # Compute loss
                loss = criterion(predicted_actions, batch_actions)

                # Backward pass and optimization step
                self.BC_optimizer.zero_grad()
                loss.backward()
                self.BC_optimizer.step()

                epoch_loss += loss.item()
            # if (epoch+1) % 20 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

        return 
      
        

    def act(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state_T = torch.from_numpy(state).float().to(self.device)
        if eval:  # test section
            action = self.actor_local.get_det_action(state_T)
            return action.numpy()

        else:  # train section

            if self.current_agent == 0:
                with torch.no_grad():
                    action = self.actor_local.get_action(state_T).numpy()

            else:
                current_social_agent = self.BC_agents_list[self.current_agent - 1] 
                
                if isinstance(current_social_agent, Agent):  # modify the SA class to remove the redundant if/else
                    action = current_social_agent.act(state, eval = True)
                else:
                    action, _ = current_social_agent.predict(state, deterministic=True)
                action = action #+ np.random.normal(0,0.01) #adding noise to actions

        return action
    
    
    
class OfflineSocialAgent(Agent):
    def __init__(self, args, social_agents_list):

        super().__init__(args)

        self.social_agents_list = social_agents_list
        self.current_agent = 0
        # self.avg_alpha = -0.1
        self.env_max_score = 200
        self.selected_agent_list = [0]
        self.n_agents = len(social_agents_list) + 1
        self.agents_normalized_scores = torch.zeros(self.n_agents)
        self.deque_len = 15
        self.scores_lists = [deque(maxlen=self.deque_len) for _ in range(self.n_agents)]

    def update_score_values(self, new_score):
        self.scores_lists[self.current_agent].append(new_score / self.env_max_score * 2)
        self.agents_normalized_scores[self.current_agent] = np.mean(self.scores_lists[self.current_agent])
            # weights=np.exp(-self.avg_alpha * np.arange(len(self.scores_lists[self.current_agent]))))       
        # self.agents_normalized_scores[0] = max(0.0, self.agents_normalized_scores[0])
        dist = Categorical(logits=self.agents_normalized_scores)
        self.current_agent = int(dist.sample())
        self.selected_agent_list.append(self.current_agent)
        return

    def act(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state_T = torch.from_numpy(state).float().to(self.device)
        if eval:  # test section
            action = self.actor_local.get_det_action(state_T)
            return action.numpy()

        else:  # train section

            if self.current_agent == 0:
                with torch.no_grad():
                    action = self.actor_local.get_action(state_T).numpy()

            else:
                current_social_agent = self.social_agents_list[self.current_agent - 1] 
                
                if isinstance(current_social_agent, Agent):  # modify the SA class to remove the redundant if/else
                    action = current_social_agent.act(state, eval = True)
                else:
                    action, _ = current_social_agent.predict(state, deterministic=True)
                action = action #+ np.random.normal(0,0.01) #adding noise to actions

        return action
    