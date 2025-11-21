import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
import sys
def clear_line_up():
    # Move cursor up one line
    sys.stdout.write("\033[1A")
    # Clear that line
    sys.stdout.write("\033[2K")
    sys.stdout.flush()
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, mask_value=0.0):
        super().__init__()
        self.mask_value = mask_value

        # Masking: we emulate this by creating a boolean mask during forward()
        # and setting masked timesteps to zero before feeding to LSTM.

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # --- Masking (similar to Keras Masking(mask_value=0.0)) ---
        # Create mask: False where masked, True otherwise
        mask = (x != self.mask_value)
        # any feature nonzero → timestep is valid
        timestep_mask = mask.any(dim=-1, keepdim=True)   # shape (B, T, 1)

        # zero-out masked positions
        x = x * timestep_mask

        # --- LSTM layers ---
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        # --- Dense layers applied per timestep ---
        out = self.relu(self.fc1(out))
        out = self.out(out)  # output shape: (batch, seq_len, output_dim)

        return out

if __name__=="__main__":
    #load in datasets
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/X_DATA.npy")[100:]
    X=(X-np.min(X))/(np.max(X)-np.min(X))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32)
    #Y_tensor = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    INPUT_DIM=X_tensor.shape[2]
    OUTPUT_DIM=12
    model = LSTMModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)
    n_epochs=100
    scheduler=np.arange(0,0.9,0.9/n_epochs)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        total_loss = 0.0
        print("Live Loss: ...")
        for (X,) in loader:     # X: (batch, seq_len, input_dim)
            X = X.to(device)
            optimizer.zero_grad()
            seq_len=X.shape[1]
            
            hidden = None                   # LSTM hidden state
            losses = []
            #predict each future step autoregressively
            window=10
            inp = X[:, :window,:]
            for t in range(window, seq_len):
                # Forward pass
                out, hidden = model.lstm1(inp, hidden)
                out, hidden = model.lstm2(out, hidden)
                pred = model.out(model.relu(model.fc1(out)))  # (batch,1,12)

                # Compute target delta
                target_delta = X[:, t-window+1:t+1, :12] - X[:, t-window:t, :12]
                target_delta = target_delta.unsqueeze(1)  # (batch,1,12)
                losses.append(criterion(pred, target_delta))

                # Reconstruct absolute motor positions
                motors = inp[:, :, :12] + pred
                # Scheduled sampling
                if np.random.random() < scheduler[epoch]:
                    remainder = X[:, t-window+1:t+1, 12:]
                    inp = torch.cat([motors.detach(), remainder.to(device)], dim=2)
                else:
                    inp = X[:, t-window+1:t+1, :]

            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            clear_line_up()
            print(f"Live Loss: {total_loss:.6f}")
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.6f} | Probability:",scheduler[epoch])
        torch.save(model.state_dict(), "/its/home/drs25/Quadruped/Code/UBERMODEL/models/lstm_gait_autoregressiveDeltasWindow.pth")
