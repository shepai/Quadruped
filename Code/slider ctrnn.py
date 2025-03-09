import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('TkAgg')
# Define the Network class
class Network:
    def __init__(self, num, std=0.05):
        self.num = num
        self.std = std
        self.A = np.random.uniform(0.1, 0.2, num) + np.random.normal(0, 0.05, num)  # Add noise
        self.weights = np.random.normal(0, std, (num, num))  # Weight matrix
        self.Tau = np.ones(num)  # Time constants
        self.dt = 0.1
        self.b = np.random.normal(0, std, num)  # Bias terms
        self.O = self.sigma(self.A)
        self.gains = np.random.normal(0, 5, (num, ))  # Gains for activation
        self.reset()
        self.bias_val=0

    def reset(self):
        self.A = np.random.uniform(0.1, 0.2, self.num) + np.random.normal(0, 0.05, self.num)
        self.O = self.sigma(self.A)

    def sigma(self, x):
        return np.tanh(x) #1 / (1 + np.exp(-x))

    def forward(self, I=0):
        self.O = self.sigma(self.gains * (self.A + self.b))
        total_inputs = np.dot(self.weights, self.O) + I
        self.A += (self.dt / self.Tau) * (total_inputs - self.A)
        
        return self.O

# Initialize the network
num_neurons = 5
network = Network(num_neurons)

# Prepare the plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
t = np.arange(0.0, 10.0, network.dt)
outputs = [network.forward()]

# Generate initial output
for _ in t:
    outputs.append(network.forward(outputs[-1].T))
outputs=outputs[0:len(t)]
print(len(t),np.array(outputs).shape)
output_plot, = ax.plot(t, np.array(outputs)[:,0])

ax.set_xlabel('Time')
ax.set_ylabel('Output')
ax.set_title('CTRNN Network Output')
ax.set_ylim([-2,2])

# Create the slider for standard deviation
ax_std = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
std_slider = Slider(ax_std, 'Std Dev', 0.01, 1.0, valinit=network.std)

# Create the slider for bias
ax_bias = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightcoral')
bias_slider = Slider(ax_bias, 'Bias', 0, 2.0, valinit=0.0)

# Function to update the network and the plot when slider changes
def update(val):
    network.std = std_slider.val
    network.weights = np.random.normal(0, network.std, (num_neurons, num_neurons))
    network.b = np.random.normal(0, bias_slider.val, num_neurons)
    network.bias_val = bias_slider.val
    network.reset()

    # Recompute outputs with the updated std
    outputs = []
    for _ in t:
        outputs.append(network.forward())

    output_plot.set_ydata(np.array(outputs)[:,0])
    fig.canvas.draw_idle()

# Connect the update function to the slider
std_slider.on_changed(update)
bias_slider.on_changed(update)

plt.show()
