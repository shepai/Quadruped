import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#load in the data



class RegressiveLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=2, output_size=60, dropout=0.1):
        super(RegressiveLSTM, self).__init__()
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer to map final hidden state → output vector
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_size]
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from last layer
        last_hidden = h_n[-1]   # shape: [batch_size, hidden_size]
        
        output = self.fc(last_hidden)
        output = torch.clamp(output, -100, 100)
        return output
if __name__=="__main__":
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_y.npy")
    y=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_X.npy")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).reshape((len(y),-1))
    print("X has NaN:", np.isnan(X).any())
    print("y has NaN:", np.isnan(y).any())
    print("X data",X.shape)
    print("y data",y.shape)
    #normalise 
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True) + 1e-8

    X = (X - mean) / std
    scaler = StandardScaler()
    y = scaler.fit_transform(y)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X_t, y_t, test_size=0.2, shuffle=True
    )
    
    #LSTM class
    #train
    model = RegressiveLSTM(input_size=11, hidden_size=64, num_layers=2, output_size=60,dropout=0.3)
    batch = 32
    seq_len = 50  # example

    def train_model(
        model,
        train_data,
        train_targets,
        val_data=None,
        val_targets=None,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        device="cpu"
    ):
        # Move model to device
        model.to(device)

        # Create datasets
        train_dataset = TensorDataset(train_data, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data is not None:
            val_dataset = TensorDataset(val_data, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        # Optimizer + Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()
        losses=[]
        # Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                preds = model(x_batch)          # [batch_size, 48]
                loss = loss_fn(preds, y_batch)  # y_batch must be [batch_size, 48]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * x_batch.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)

                        preds = model(x_val)
                        loss = loss_fn(preds, y_val)
                        val_loss += loss.item() * x_val.size(0)

                val_loss /= len(val_loader.dataset)
                print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

            else:
                print(f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f}")
            losses.append([epoch_loss,val_loss])
            np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/models/regressive_lstm_loss_norm_t3",np.array(losses))
            torch.save(model.state_dict(), "/its/home/drs25/Quadruped/Code/UBERMODEL/models/regressive_lstm_norm_t3.pth")

    train_model(
        model,
        train_data=X_train,
        train_targets=y_train,
        val_data=X_val,
        val_targets=y_val,
        epochs=1500,
        batch_size=32,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
