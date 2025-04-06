import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer, HookedTransformerConfig
from tqdm.notebook import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)



if __name__ == "__main__":
    print("Hello, world!")