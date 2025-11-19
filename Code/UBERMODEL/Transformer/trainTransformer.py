import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from dataclasses import dataclass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output = nn.Linear(model_dim, output_dim)

        # Optional learnable positional encoding
        self.pos = nn.Parameter(torch.randn(1, T, model_dim) * 0.01)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos[:, :x.size(1)]
        x = self.encoder(x)
        return self.output(x)      # (B, T, V)

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
    return total_loss / total_samples
class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__=="__main__":
    #load in the data 
    N = 100     # number of sequences
    T = 200     # timesteps
    V1 = 20      # joints
    V2 = 12      # joints
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #X, y = generate_noisy_sine_dataset(N, T, V1,V2, noise_std=0.2)
    X=torch.tensor(np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA.npy"),  dtype=torch.float32)[100:]
    y=torch.tensor(np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/y_DATA.npy"),  dtype=torch.float32)[100:]
    print(X.shape,y.shape)
    dataset = MotorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    n_epochs = 1500
    train_losses = []
    val_losses = []
    model = MotorTransformer(input_dim=V1,output_dim=V2,T=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            pred = model(xb.to(device))
            loss = loss_fn(pred, yb.to(device))

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")
    torch.save(model.state_dict(), "/its/home/drs25/Quadruped/Code/UBERMODEL/models/gait_transformer.pth")
    