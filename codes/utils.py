import torch
import numpy as np
from torch.utils.data import Dataset
from itertools import count
import gymnasium as gym
def evaluate(env, policy, eval_runs=20): 
    """
    Makes an evaluation run with the current policy
    """
    scores_list = []
    for i in range(eval_runs):
        state, _ = env.reset()

        score = 0
        while True:
            # state = state.reshape((1, policy.state_size))
            action = policy.act(state, eval=True)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            if done:
                break
        scores_list.append(score)
    return np.mean(scores_list)



# Define a custom PyTorch Dataset
class ExpertDataset(Dataset):
    def __init__(self, transitions, device = 'cuda'):
      
      state_dim = transitions[0]['obs'].shape[0]
      action_dim = transitions[0]['acts'].shape[0]
      data_len = len(transitions)
      
      self.states = torch.zeros((data_len, state_dim))
      self.actions = torch.zeros((data_len, action_dim))
      
      for i in range(data_len):
        
        self.states[i] = torch.tensor(transitions[i]['obs'], dtype=torch.float32)
        self.actions[i] = torch.tensor(transitions[i]['acts'], dtype=torch.float32)
      self.states = self.states.to(device)
      self.actions = self.actions.to(device)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    
def evaluate_agent(agent, n_eval = 100):

    reward_list_eval = []
    # agent.actor_local.eval()
    env = gym.make('BipedalWalker-v3', max_episode_steps= 750, hardcore = True, render_mode = None)

    for i_test in range(n_eval):
        # agent.current_agent = 2
        state, _ = env.reset()
        score = 0
        for j_test in count():
            action, _ = agent.predict(state=state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward

            if done:
                break 
        reward_list_eval.append(score)
        # print(f"episode {i_test+1} Scored {score:0.2f} in {j_test+1} steps")
    env.close()
    return np.mean(reward_list_eval), np.std(reward_list_eval)


def print_time_format(t0, t1):
    
                
    elapsed_time = t1 - t0

    # Convert elapsed time to hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Print the results in hh:mm:ss format
    print(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")