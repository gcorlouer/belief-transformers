import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm.notebook import tqdm
#TODO: 
# -check correctness of generated process
# -check stationary distribution is correct
# -maybe need to renormalize probabilities (belief states and from transition matrices)
# -Include type of variables 

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Mess3Process():
    def __init__(self):
        # Define the transition matrices for Mess3 as given in the paper
        self.T_A = torch.tensor([
            [0.765, 0.00375, 0.00375],
            [0.0425, 0.0675, 0.00375],
            [0.0425, 0.00375, 0.0675]
        ])
        
        self.T_B = torch.tensor([
            [0.0675, 0.0425, 0.00375],
            [0.00375, 0.765, 0.00375],
            [0.00375, 0.0425, 0.0675]
        ])
        
        self.T_C = torch.tensor([
            [0.0675, 0.00375, 0.0425],
            [0.00375, 0.0675, 0.0425],
            [0.00375, 0.00375, 0.765]
        ])

        self.T = torch.stack([self.T_A, self.T_B, self.T_C], dim=0)

        # Token map
        self.tokens = ['A', 'B', 'C']
        self.states = [0, 1, 2]
        self.token_idx = {token: idx for idx, token in enumerate(self.tokens)}

    def stationary_distribution(self):
        # Compute the stationary distribution of the Markov chain
        # Solve the eigenvector problem: T * pi = pi
        eig_vals, eig_vecs = torch.linalg.eig(self.T.sum(dim=0))
        eig_vals = eig_vals.real
        eig_vecs = eig_vecs.real
        stationary_distribution = eig_vecs[:, torch.isclose(eig_vals, torch.tensor(1.0))]
        pi = stationary_distribution / stationary_distribution.sum()
        return pi
    
    def generate_sequence(self, length):
        start_state = np.random.choice(self.states, p=self.stationary_distribution().squeeze().numpy())
        state_sequence = [start_state]
        tok_sequence = []
        current_state = start_state
        for _ in range(length):
            token = np.random.choice(self.tokens, p=self.T[:, :, current_state].sum(dim=1).squeeze().numpy())
            tok_sequence.append(token.item())
            next_state = np.random.choice(self.states, p=self.T[:, :, current_state].sum(dim=0).squeeze().numpy())
            state_sequence.append(next_state)
            current_state = next_state
        return tok_sequence

    def get_belief_states(self, sequence):
        belief_states = []
        belief_state = self.stationary_distribution()
        for token in sequence:
            token_idx = self.token_idx[token]
            belief_state = torch.einsum('i,ij->j', belief_state, self.T[token_idx, :, :])
            belief_state = belief_state / belief_state.sum()
            belief_states.append(belief_state)
        return belief_states
    
def plot_belief_states(belief_states):
    

if __name__ == "__main__":
    mess3 = Mess3Process()
    pi = mess3.stationary_distribution()
    sequence = mess3.generate_sequence(100)
    print(sequence)