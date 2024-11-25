import numpy as np
from copy import deepcopy
from agent import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network:
    def __init__(self, num, std=0.1):
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
"""class Network:
    def __init__(self,num,std=5):
        self.A=np.zeros((num))
        self.weights=np.random.normal(0,std,(num,num))
        np.fill_diagonal(self.weights, 0) #zero self weights
        self.Tau=1
        self.dt=0.01
        self.b=0
    def sigma(self,x):
        return np.tanh(x)#1/(1+np.exp(-x))
    def forward(self,I=0):
        #TA=-self.A.T + np.dot(self.act(self.A.T),self.weights)+ I.T +self.bias #forward pass
        O=self.sigma(self.A+self.b)
        self.A = self.A + (self.dt/self.Tau)*(-self.A + np.dot(self.weights,O)+I)
        return self.A"""
    
class CPG(agent):
    def __init__(self,num_neurons):
        agent.__init__(self)
        self.cpg=Network(num_neurons)
        self.num_neurons=num_neurons
        self.geno=np.concatenate([self.cpg.weights.flatten(),self.cpg.b.flatten(),self.cpg.Tau.flatten()])
        self.set_genotype(self.geno)
        self.populateBody()
    def populateBody(self):
        self.body=[]
        for i in range(4): #one for each leg
            self.body.append(deepcopy(self.cpg))
    def get_positions(self,inputs):
        positions=[]
        Inputs=np.zeros((self.num_neurons))
        #Inputs[0]+=inputs[0] #add proprioception
        #Inputs[1]+=inputs[1]
        #Inputs[2]+=inputs[2]
        for cpg in self.body:
            out=cpg.forward(I=Inputs) #forward
            Inputs=out
            Inputs[0]+=inputs[0]
            Inputs[1]+=inputs[1]
            Inputs[2]+=inputs[2]
            #Inputs=self.cpg.sigma(Inputs)*15
            Inputs[Inputs<-0]=-0
            Inputs[Inputs>40]=40
            positions.append(Inputs[0]) #select neuron outputs as proportional to motor
            positions.append(Inputs[1])
            positions.append(Inputs[2])

        #print(np.degrees(np.array(positions))/5)
        return np.array(positions)
    def set_genotype(self, values):
        self.cpg.weights=values[0:len(self.cpg.weights.flatten())].reshape(self.cpg.weights.shape)
        self.cpg.b=values[len(self.cpg.weights.flatten()):len(self.cpg.weights.flatten())+len(self.cpg.b.flatten())]
        self.cpg.b[self.cpg.b<-4] = -4#cap bias
        self.cpg.b[self.cpg.b>4] = 4 #cap bias
        self.cpg.Tau=values[len(self.cpg.weights.flatten())+len(self.cpg.b.flatten()):len(self.cpg.weights.flatten())+len(self.cpg.b.flatten())+len(self.cpg.Tau)].reshape(self.cpg.Tau.shape)
        self.cpg.weights[self.cpg.weights>16]=16 #cap weights
        self.cpg.weights[self.cpg.weights<-16]=-16 #cap weights
        self.populateBody()
        return super().set_genotype(np.concatenate([self.cpg.weights.flatten(),self.cpg.b.flatten(),self.cpg.Tau.flatten()]))
    
class Pattern:
    def __init__(self,a,h,b,k):
        self.geno=torch.tensor([a,h,b,k])
    def forward(self,t):
        g=self.geno
        return g[0] * torch.sin((t-g[1])/g[2]) +g[3]
    def set_param(self,a,h,b,k):
        self.geno=torch.tensor([a,h,b,k])

class generator(agent):
    def __init__(self):
        agent.__init__(self)
        g1=np.random.normal(0,1,(4))
        g2=np.random.normal(0,1,(4))
        hips=np.random.normal(0,1,(4))
        phase=np.array([np.random.randint(0,100) for i in range(4)])
        self.leg=Pattern(*g1)
        self.knee=Pattern(*g2)
        self.hip=Pattern(*hips)
        self.geno=np.concatenate([phase,g1,g2,hips])
        self.val=0
    def get_positions(self,inputs):
        positions=[]
        #Inputs[0]+=inputs[0] #add proprioception
        #Inputs[1]+=inputs[1]
        #Inputs[2]+=inputs[2]
        for i in range(4):
            phase=self.geno[i]
            hip=self.hip.forward(phase+self.val) #forward
            leg=self.leg.forward(phase+self.val) #forward
            knee=self.knee.forward(phase+self.val) #forward
            positions.append(hip)
            positions.append(leg)
            positions.append(knee)
        self.val+=1
        #print(np.degrees(np.array(positions))/5)
        positions=(np.array(positions)/5)*180
        positions[positions<0]=0
        positions[positions>180]=180
        return np.degrees(positions)
    def set_genotype(self,val):
        #mutate instead
        self.geno=val.copy()
        self.leg=Pattern(*self.geno[4:8])
        self.knee=Pattern(*self.geno[8:12])
        self.hip=Pattern(*self.geno[12:16])
        self.val=0
    def mutate(self):
        phase=np.array([np.random.randint(-50,50) for i in range(4)])
        self.geno[0:4]+=phase
        self.geno[0:4][self.geno[0:4]>100]=100
        self.geno[0:4][self.geno[0:4]<0]=0
        self.geno[4:]+=np.random.normal(0,1,self.geno[4:].shape)

import torch
import torch.nn as nn
import numpy as np

class NN(nn.Module):
    def __init__(self, inp, hidden,env=0):
        super(NN, self).__init__()
        out = 16  # Output size
        self.fc1 = nn.Linear(inp, hidden)  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden, out)  # Fully connected layer 2
        self.genotype = self._get_genotype()  # Flattened parameter array
        self.sig = Pattern(0, 0, 0, 0)  # External object, assumed defined elsewhere
        self.val = 0  # Keeps track of the step for position encoding

    def _get_genotype(self):
        """Flatten all model parameters into a single array."""
        with torch.no_grad():
            params = []
            for param in self.parameters():
                params.append(param.view(-1).cpu().numpy())
            return np.concatenate(params)

    def _set_genotype(self, genotype):
        """Set model parameters from a flattened array."""
        with torch.no_grad():
            idx = 0
            for param in self.parameters():
                numel = param.numel()
                param.copy_(torch.tensor(genotype[idx:idx + numel]).view(param.shape))
                idx += numel

    def forward(self, X):
        """Forward pass through the network."""
        x = torch.sigmoid(self.fc1(X))  # Apply sigmoid activation
        x = self.fc2(x)  # Pass through the second layer
        self.cache = {'X': X, 'hidden': x}  # Cache for backpropagation
        return x / 3  # Scale the output
    def forward_positions(self,x,motors=0):
        positions = np.zeros(12)
        for j, i in enumerate(range(0, 12, 3)):  # Loop through legs assigning phase encoding
            self.sig.set_param(*x[0:4])
            positions[i] = self.sig.forward(x[j] + self.val)
            self.sig.set_param(*x[4:8])
            positions[i + 1] = self.sig.forward(x[j] + self.val)
            self.sig.set_param(*x[8:12])
            positions[i + 2] = self.sig.forward(x[j] + self.val)
        
        self.val += 1
        positions = motors + ((positions)/5)*50
        positions[positions < 0] = 0
        positions[positions > 50] = 50
        return torch.tensor(positions)
    def get_positions(self, x, motors=0):
        """Generate positions for the robot's motors."""
        if motors is None:
            motors = torch.zeros(12)  # Default to zeros if not provided

        if len(x.shape) < 2:
            x = x.unsqueeze(0)  # Ensure input is 2D for batch processing

        x = self.forward(x)[0]  # Process input and convert to numpy
        positions=self.forward_positions(x)
        return positions

    def set_genotype(self, genotype):
        """Set genotype and update model parameters."""
        self.genotype = genotype
        self._set_genotype(self.genotype)

    def mutate(self, std=2.0):
        """Mutate the model's parameters by adding noise."""
        mutated_genotype = self.genotype + np.random.normal(0, std, self.genotype.shape)
        self.set_genotype(mutated_genotype)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical


class CustomNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Initialize the base ActorCriticPolicy
        super(CustomNNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        
        # Define the custom neural network
        self.net = NN(inp=observation_space.shape[0], hidden=64)
        self.sig = Pattern(0, 0, 0, 0)  # External object, assumed defined elsewhere
        self.val = 0  # Keeps track of the step for position encoding

    def forward(self, obs, *args, **kwargs):
        """Forward pass through policy and value networks."""
        print(">>>",obs)
        print(obs.actor)
        obs=torch.tensor(obs.sample(),dtype=torch.float32)
        features = self.net(obs)  # Pass observation through the custom NN
        actions=self.forward_positions(features)
        self.val+=1
        return actions
    def forward_positions(self,x,motors=0):
        positions = np.zeros(12)
        for j, i in enumerate(range(0, 12, 3)):  # Loop through legs assigning phase encoding
            self.sig.set_param(*x[0:4])
            positions[i] = self.sig.forward(x[j] + self.val)
            self.sig.set_param(*x[4:8])
            positions[i + 1] = self.sig.forward(x[j] + self.val)
            self.sig.set_param(*x[8:12])
            positions[i + 2] = self.sig.forward(x[j] + self.val)
        
        self.val += 1
        positions = motors + ((positions)/5)*50
        positions[positions < 0] = 0
        positions[positions > 50] = 50
        return torch.tensor(positions)
    def evaluate_actions(self, obs, actions):
        """Evaluate the actions for the given observations."""
        features = self.net(obs)
        actions=self.forward_positions(features)
        
        distribution = Categorical(logits=actions)
        log_probs = distribution.log_prob(features)
        entropy = distribution.entropy()
        
        return actions, log_probs, entropy

    def get_positions(self, obs, motors=0):
        """Generate positions for the robot's motors."""
        positions = self.net.get_positions(obs, motors=motors)
        return positions

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    """from environment import *
    env=environment(True)
    agent=CPG(3)
    env.runTrial(agent,1000,True)"""
    """gen=generator()
    
    positions=[]
    for i in range(100):
        pos=gen.get_positions(0)
        positions.append(pos)
    positions=np.array(positions)
    print(positions)
    c=["b","r","g","b","r","g","b","r","g","b","r","g"]
    label=["hip1","knee1","foot1","hip2","knee2","foot2","hip3","knee3","foot3","hip4","knee4","foot4"]
    for i in range(12):
        plt.plot(positions[:,i],c=c[i],label=label[i])
    plt.legend(loc="upper right")
    plt.show()"""
    n=NN(10,20)
    n.set_genotype(torch.rand(n.genotype.shape))
    motors=[]
    for i in range(300):
        motors.append(n.get_positions(torch.zeros((1,10))).detach().numpy())
    plt.plot(np.array(motors))
    plt.show()
    



