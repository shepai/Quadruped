import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32)
    #Y_tensor = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    INPUT_DIM=X_tensor.shape[2]
    OUTPUT_DIM=12
    model = LSTMModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        total_loss = 0.0
        for (X,) in loader:     # X: (batch, seq_len, input_dim)
            X = X.to(device)
            optimizer.zero_grad()
            seq_len=X.shape[1]
            inp = X[:, 0].unsqueeze(1) 
            hidden = None                   # LSTM hidden state
            losses = []
            remainder = X[:, 0:1, 12:]     # shape (1,1,4)
            #predict each future step autoregressively
            for t in range(seq_len - 1):
                out, hidden = model.lstm1(inp, hidden)   # (B,1,128)
                out, hidden = model.lstm2(out, hidden)
                pred = model.out(model.relu(model.fc1(out)))  # (B,1,D)

                # target is the NEXT real timestep
                target = X[:, t+1,:12].unsqueeze(1)

                # loss on this step
                losses.append(criterion(pred, target))

                # FEED PREDICTION BACK IN
                #print(pred.shape,remainder.shape)
                if np.random.random()<0.3: #30%
                    inp = torch.concatenate([pred.detach(),remainder.to(device)], axis=2)   # freeze gradients through the input
                else:
                    inp = X[:,t+1].unsqueeze(1) 
                    remainder = X[:, t+1:t+2, 12:]
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), "/its/home/drs25/Quadruped/Code/UBERMODEL/models/lstm_gait_autoregressive.pth")
