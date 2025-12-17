import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

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
        return output



def reform(height, gap, width, level, start, repeat=1):
    #rising/peak part sampled at integer indices
    x1 = np.arange(start,start+gap)
    y1 = height * np.sin((np.pi / gap) * (x1 - start))
    x2 = np.arange(start+gap, start+gap+width)#flat base part
    y2 = np.ones_like(x2) * level
    x = np.concatenate((x1, x2))#one full cycle
    y = np.concatenate((y1, y2))
    cycles = []#repeat cycles exactly
    ys = []
    for i in range(repeat):
        offset = i * (gap + width)
        cycles.append(x + offset)
        ys.append(y)
    x = np.concatenate(cycles)
    y = np.concatenate(ys)
    # shift so first peak is at index "start"
    peaks, _ = find_peaks(np.abs(y))
    shift = peaks[0] - start #find difference between our generated signal
    x-=start
    y = np.roll(y, -shift) #roll it round so it alignes with the start position
    return x, y
if __name__=="__main__":
    import matplotlib 
    matplotlib.use('TkAgg')
    #test 
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = RegressiveLSTM(input_size=11, hidden_size=64, num_layers=2, output_size=60).to(device)

    model.load_state_dict(torch.load("/its/home/drs25/Quadruped/Code/UBERMODEL/models/regressive_lstm_norm.pth", map_location=device))
    model.eval()
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_y.npy")
    y=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_X.npy")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).reshape((len(y),-1))
    print("X has NaN:", np.isnan(X).any())
    print("y has NaN:", np.isnan(y).any())
    #normalise 
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True) + 1e-8

    X = (X - mean) / std
    scaler = StandardScaler()
    y = scaler.fit_transform(y)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape((len(y),-1))
    X_train, X_val, y_train, y_val = train_test_split(
        X_t, y_t, test_size=0.2, shuffle=True
    )
    print("X data",X.shape)
    print("y data",y.shape)
    #LSTM class
    idx = np.random.randint(0,len(X_t))
    x_sample = X_t[idx]           # shape: [seq_len, 11]
    y_true = scaler.inverse_transform(y)
    y_true = y_true[idx]

    x_sample = x_sample.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_sample.to(device)).cpu()
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = y_pred.squeeze(0).flatten()
    cmap = plt.cm.tab20

    plt.figure(figsize=(8,6))

    for idx, i in enumerate(range(0, 60-5, 5)):
        # Choose a base colour for this iteration
        base_color = cmap(idx / 12)   # 12 iterations for range(0,55,5)
        # Make a lighter version for the predicted signal
        lighter_color = mcolors.to_rgba(base_color, alpha=0.5)
        # Take params
        params_m1 = y_true.flatten()[i:i+5]
        params_m2 = y_pred[i:i+5]
        x1, y1 = reform(*params_m1)
        x2, y2 = reform(*params_m2)
        # Plot original (use zorder to keep lines visible)
        plt.plot(
            x1, y1, 
            color=base_color, 
            linewidth=2,
            label="Original signal" if idx == 0 else "_nolegend_"
        )
        # Plot predicted
        plt.plot(
            x2, y2, 
            color=lighter_color, 
            linestyle="--", 
            linewidth=2,
            label="Predicted signal" if idx == 0 else "_nolegend_"
        )

    plt.title("Model prediction vs truth of motor prediction", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()