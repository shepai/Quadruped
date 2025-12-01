import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
def clear_line_up():
    # Move cursor up one line
    sys.stdout.write("\033[1A")
    # Clear that line
    sys.stdout.write("\033[2K")
    sys.stdout.flush()
"""SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)"""
def generate_noisy_sine_dataset(N, T, V1,V2, noise_std=0.1):
    """
    N = number of sequences
    T = timesteps
    V = joints
    """
    X = torch.zeros((N, T, V1))
    y = torch.zeros((N, T, V2))

    for n in range(N):
        t = np.linspace(0, 4*np.pi, T+1)  # +1 for next-step target

        # Make each joint have its own amplitude, frequency, phase, offset
        amp   = np.random.uniform(0.5, 2.5, size=V1)
        freq  = np.random.uniform(0.5, 2.0, size=V1)
        phase = np.random.uniform(0, 2*np.pi, size=V1)
        off   = np.random.uniform(-1.0, 1.0, size=V1)

        # Build the signal
        signal = np.zeros((T+1, V1))
        for j in range(V1):
            signal[:, j] = off[j] + amp[j] * np.sin(freq[j] * t + phase[j])

        # Add noise
        signal += np.random.normal(scale=noise_std, size=signal.shape)

        # Fill X[t] = signal[t], y[t] = signal[t+1]
        X[n] = torch.tensor(signal[:-1], dtype=torch.float32)
        amp   = np.random.uniform(0.5, 2.5, size=V2)
        freq  = np.random.uniform(0.5, 2.0, size=V2)
        phase = np.random.uniform(0, 2*np.pi, size=V2)
        off   = np.random.uniform(-1.0, 1.0, size=V2)

        # Build the signal
        signal = np.zeros((T+1, V2))
        for j in range(V2):
            signal[:, j] = off[j] + amp[j] * np.sin(freq[j] * t + phase[j])

        # Add noise
        signal += np.random.normal(scale=noise_std, size=signal.shape)
        y[n] = torch.tensor(signal[1:],  dtype=torch.float32)

    return X, y
class MotorTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, T,model_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            batch_first=True
        )
        self.register_buffer("pos_encoding", self.create_sinusoidal_encoding(T, model_dim))

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output = nn.Linear(model_dim, output_dim)

        # Optional learnable positional encoding
        self.pos = nn.Parameter(torch.randn(1, T, model_dim) * 0.01)
    def create_sinusoidal_encoding(self, T, dim):
        """Return (T, dim) sinusoidal positional encoding."""
        pe = torch.zeros(T, dim)
        position = torch.arange(0, T).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions

        return pe.unsqueeze(0)   # shape: (1, T, dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos[:, :x.size(1)]
        x = self.encoder(x)
        return self.output(x)      # (B, T, V)


if __name__=="__main__":
    #load in the data 
    N = 100     # number of sequences
    T = 200     # timesteps
    V2 = 12      # joints
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA_one.npy")[0:1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean=np.mean(X)
    std=np.std(X)
    X = (X - mean) / std
    X_tensor = torch.tensor(X, dtype=torch.float32)
    V1 = X[:1,:,:].shape[2]      # joints
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    n_epochs = 1000
    scheduler=np.arange(0,0.9,0.9/n_epochs)
    train_losses = []
    val_losses = []
    model = MotorTransformer(input_dim=V1,output_dim=V2,T=T).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        print("Live Loss: ...")
        for (X,) in loader:     # X: (batch, seq_len, input_dim)
            X = X.to(device)
            optimizer.zero_grad()
            seq_len=X.shape[1]
            inp = X[:, 0].unsqueeze(1) 
            hidden = None                   # LSTM hidden state v  
            losses = []
            #predict each future step autoregressively
            window=15
            inp = X[:, :window,:]
            for t in range(window, seq_len):
                # Forward pass
                pred = model(inp)  # (batch,1,12)
                target=X[:, t:t+1, :12] #just get the next key
                #target_delta = X[:, t-window+1:t+1, :12] - X[:, t-window:t, :12]
                #target_delta = target_delta.unsqueeze(1)  # (batch,1,12)
                losses.append(criterion(pred[:,-1:,:], target))
                motors = pred #inp[:, :, :12] + pred
                #scheduled sampling though i have commented this out
                if np.random.random() < scheduler[epoch] and False:
                    remainder = X[:, t-window+1:t+1, 12:]
                    inp = torch.cat([motors.detach(), remainder.to(device)], dim=2)
                else:
                    remainder = X[:, t-window+1:t+1, :]
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            clear_line_up()
            print(f"Live Loss: {total_loss:.6f}")
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.6f} | Probability:",scheduler[epoch])
        torch.save(model.state_dict(), "/its/home/drs25/Quadruped/Code/UBERMODEL/models/transformer_gait_one1.pth")
    