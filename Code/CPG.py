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
        self.weights = np.random.normal(0, std, (num, num))# np.random.random(0, std, (num, num))  # Weight matrix
        self.Tau = np.ones(num)  # Time constants
        self.dt = 0.1
        self.b = np.random.normal(0, std, num)#np.random.normal(0, std, num)  # Bias terms
        self.O = self.sigma(self.A)
        self.gains = np.ones((num,))#np.random.normal(0, 1, (num, ))  # Gains for activation
        #self.gains[self.gains>2]=2
        #self.gains[self.gains>-2]=-2
        self.reset()
        self.bias_val=0
        self.t=0
    def reset(self):
        self.A = np.random.uniform(0.1, 0.2, self.num) + np.random.normal(0, 0.05, self.num)
        self.O = self.sigma(self.A)
        self.t = 0
    def sigma(self, x):
        return 1 / (1 + np.exp(-x)) #np.tanh(x) #

    def forward(self, I=0):
        I+=np.sin(self.t)
        self.O = self.sigma(self.gains * (self.A + self.b))
        #print(self.weights.shape,self.O.shape,I.shape)
        total_inputs = np.dot(self.weights, self.O) + I
        self.A += (self.dt / self.Tau) * (total_inputs - self.A)
        self.t+=self.dt
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
    def get_positions(self,inputs,motors=None):
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
        self.cpg.b[self.cpg.b<-16] = -16#cap bias
        self.cpg.b[self.cpg.b>16] = 16 #cap bias
        self.cpg.Tau=values[len(self.cpg.weights.flatten())+len(self.cpg.b.flatten()):len(self.cpg.weights.flatten())+len(self.cpg.b.flatten())+len(self.cpg.Tau)].reshape(self.cpg.Tau.shape)
        self.cpg.Tau[self.cpg.Tau<self.cpg.dt]=self.cpg.dt
        self.cpg.weights[self.cpg.weights>4]=4 #cap weights
        self.cpg.weights[self.cpg.weights<-4]=-4 #cap weights
        #self.populateBody()
        return super().set_genotype(np.concatenate([self.cpg.weights.flatten(),self.cpg.b.flatten(),self.cpg.Tau.flatten()]))
    def mutate(self,rate=0.2):
        probailities=np.random.random(self.geno.shape)
        self.geno[np.where(probailities<rate)]+=np.random.normal(0,4,self.geno[np.where(probailities<rate)].shape)
        self.set_genotype(self.geno)

class Body(CPG):
    def __init__(self,num_neurons,num_legs):
        CPG.__init__(self,num_neurons)
        self.num_legs=num_legs
        self.cpg=Network(num_neurons)
        self.legs=[deepcopy(self.cpg) for i in range(num_legs)]
        self.num_neurons=num_neurons
        self.geno=np.concatenate([self.cpg.weights.flatten(),self.cpg.b.flatten(),self.cpg.Tau.flatten()])
        self.set_genotype(self.geno)
        self.populateBody()
    def get_positions(self,motors,amount=12):
        positions=np.zeros((amount,))
        i=0
        #flip=True
        previous=np.zeros((self.num_neurons,)).astype(np.float64)
        for genorator in self.legs:
            out=genorator.forward(I=previous)
            positions[0+i:3+i]=np.degrees(out[0:3]*1.5)
            #if flip: flip = not flip
            #else: positions[0+i:3+i]*=-1 #flip the legs for symmetry
            i+=3
            previous=np.concatenate([[0.0 for j in range(self.num_neurons-3)],motors[0+(i-3):3+(i-3)]/100]).astype(np.float64)
        positions[positions<0]=0.0
        positions[positions>180]=180.0
        #print(positions)
        return positions
    def mutate(self,rate=0.2):
        probailities=np.random.random(self.geno.shape)
        self.geno[np.where(probailities<rate)]+=np.random.normal(0,4,self.geno[np.where(probailities<rate)].shape)
        self.set_genotype(self.geno)
        self.legs=[deepcopy(self.cpg) for i in range(self.num_legs)]


class sinBot:
    def __init__(self, num_legs=4, num_motors_per_leg=3, dt=0.1,imu=False):
        self.num_legs = num_legs
        self.num_motors = num_legs * num_motors_per_leg  # 12 motors total
        self.dt = dt  # Time step for integration
        self.hip_geno=np.random.uniform(-1,1,(3))
        self.leg_geno=np.random.uniform(-1,1,(3))
        self.phase=np.random.uniform(-1,1,(4))
        self.geno=np.concatenate([self.hip_geno,self.leg_geno,self.phase])
        self.t=0
    def get_positions(self,inputs,motors=None):
        degrees=np.degrees(self.step(imu_feedback=inputs, velocity_feedback=0))/1.5
        degrees=np.clip(degrees,0,180)
        degrees[[2,5,8,11]]=degrees[[1,4,7,10]]
        degrees[3:9]=-degrees[3:9] #try running this 
        return degrees
    def step(self, imu_feedback, velocity_feedback=0):
        motor_positions=[]
        for i in range(self.num_legs):
            H=self.hip_geno[0]*np.sin(self.hip_geno[1]*self.t - self.phase[i]) + self.hip_geno[2]
            L=self.leg_geno[0]*np.sin(self.leg_geno[1]*self.t - self.phase[i]) + self.leg_geno[2]
            motor_positions.append([H,L,0])
        self.t+=self.dt
        return np.array(motor_positions).flatten()
    def mutate(self,rate=0.2):
        probailities=np.random.random(self.geno.shape)
        self.geno[np.where(probailities<rate)]+=np.random.normal(0,4,self.geno[np.where(probailities<rate)].shape)
        self.set_genotype(self.geno)
    def sex(self,geno1,geno2,prob_winning=0.6):
        probabilities=np.random.random(len(self.geno))
        geno2.geno[np.where(probabilities<prob_winning)]=geno1.geno[np.where(probabilities<prob_winning)]
        geno2.set_genotype(geno2.geno)
        return geno2
    def set_genotype(self, values):
        self.t=0
        self.hip_geno=values[0:4]
        self.leg_geno=values[4:8]
        self.phase=values[8:]
        self.hip_geno=np.clip(self.hip_geno,-4,4)
        self.leg_geno=np.clip(self.leg_geno,-4,4)
        self.phase=np.clip(self.hip_geno,-1,1)
class CTRNNQuadruped:
    def __init__(self, num_legs=4, num_motors_per_leg=3, dt=0.1,imu=False):
        self.num_legs = num_legs
        self.num_motors = num_legs * num_motors_per_leg  # 12 motors total
        self.dt = dt  # Time step for integration
        self.num_neurons=6
        #initialize CTRNN parameters
        self.tau = np.ones(self.num_neurons) * 0.5  # Time constants (modifiable via evolution)
        self.weights = np.random.uniform(-1, 1, (self.num_neurons, self.num_neurons))  # Synaptic weights .normal(0,2,(self.num_neurons, self.num_neurons))#
        self.biases = np.zeros(self.num_neurons)  # Bias terms
        self.activations = np.zeros(self.num_neurons)  # Neuron activations
        self.outputs = np.zeros(self.num_neurons)  # Motor output (joint angles)

        #frequency and phase offsets for oscillatory behavior
        self.omega = np.random.uniform(0.8, 1.2, self.num_motors)  # Oscillation frequencies
        self.phases = np.linspace(0, 2 * np.pi, self.num_motors, endpoint=False)  # Initial phase

        self.geno=np.concatenate([self.weights.flatten(),self.biases.flatten(),self.tau.flatten(),self.omega.flatten()])

        #IMU Feedback Gains (Proportional control for stability)
        self.Kp_imu = 0.5  # Adjusts hip based on tilt
        self.Kp_vel = 0.3  # Adjusts knee based on forward velocity
        self.height=1
        self.imu=imu
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    def get_positions(self,inputs,motors=None):
        degrees=np.degrees(self.step(imu_feedback=inputs, velocity_feedback=0))/1.5
        degrees=np.clip(degrees,0,180)
        degrees[[2,5,8,11]]=degrees[[1,4,7,10]]
        degrees[3:9]=-degrees[3:9] #try running this 
        return degrees
    def step(self, imu_feedback, velocity_feedback):
        """Update the CTRNN for one timestep."""
        imu_feedback = np.array(imu_feedback).flatten()
        # Create fixed input weights if not already done (move this to __init__ ideally)
        input_weights = np.random.uniform(-1, 1, (self.num_neurons, 3)) if self.imu else np.zeros((self.num_neurons, 3))

        # Convert imu_feedback to a numpy array if it's not already
        imu_feedback = np.array(imu_feedback)

        # Project IMU feedback into neuron space
        sensory_drive = input_weights @ imu_feedback  # shape: (num_neurons,)

        # Compute neural activations (discrete update of CTRNN)
        net_input = self.weights @ self.outputs + self.biases + sensory_drive
        net_input = np.clip(net_input, -500, 500)
        self.activations += self.dt / self.tau * (-self.activations + net_input)
        self.activations = np.nan_to_num(self.activations, nan=0.0, posinf=1.0, neginf=-1.0)
        self.outputs = self.sigmoid(self.activations)

        # Add oscillatory gait modulation
        self.phases += self.dt * self.omega
        oscillation = np.sin(self.phases)

        # Compute motor commands (combine CTRNn output and oscillation)
        motor_commands = np.concatenate([self.outputs[0:3]] * 4) + 0.5 * oscillation
        return np.clip(motor_commands, 0, 1)*self.height  # Return motor positions (normalized)
    def set_genotype(self, values):
        """Set CTRNN parameters from an evolutionary genotype."""
        num_weights = len(self.weights.flatten())
        num_biases = len(self.biases.flatten())
        num_tau = len(self.tau.flatten())
        num_omega = len(self.omega.flatten())
        #assign genotype values to weights, biases, and time constants
        self.weights = values[0:num_weights].reshape(self.weights.shape)
        self.biases = values[num_weights:num_weights + num_biases]
        self.tau = values[num_weights + num_biases:num_weights + num_biases + num_tau].reshape(self.tau.shape)
        self.omega = values[num_weights + num_biases + num_tau: num_weights + num_biases + num_tau + num_omega].reshape(self.omega.shape)
        #apply value constraints
        self.biases = np.clip(self.biases, -16, 16)  # Cap bias values
        self.tau = np.maximum(self.tau, self.dt)  # Ensure time constants are above dt
        self.weights = np.clip(self.weights, -4, 4)  # Cap weight values
        self.omega = np.clip(self.omega, -1, 1)  # Cap weight values
        self.geno=np.concatenate([self.weights.flatten(),self.biases.flatten(),self.tau.flatten(),self.omega.flatten()])
    def mutate(self,rate=0.2):
        probailities=np.random.random(self.geno.shape)
        self.geno[np.where(probailities<rate)]+=np.random.normal(0,4,self.geno[np.where(probailities<rate)].shape)
        self.set_genotype(self.geno)
    def sex(self,geno1,geno2,prob_winning=0.6):
        probabilities=np.random.random(len(self.geno))
        geno2.geno[np.where(probabilities<prob_winning)]=geno1.geno[np.where(probabilities<prob_winning)]
        geno2.set_genotype(geno2.geno)
        return geno2
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
    def get_positions(self,inputs,motors=None):
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
        return positions
    def set_genotype(self,val):
        #mutate instead
        self.geno=val.copy()
        self.leg=Pattern(*self.geno[4:8])
        self.knee=Pattern(*self.geno[8:12])
        self.hip=Pattern(*self.geno[12:16])
        #self.val=0
    def reset(self):
        self.val=0
    def mutate(self,rate=0.1):
        phase=np.array([np.random.normal(0,5) for i in range(4)])
        probs=np.random.random(phase)
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
        self.inp=inp
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
    def sample(self):
        x=torch.rand(1,self.inp)
        x=self.forward(x)
        x=self.forward_positions(x)
        return x
    def forward(self, X):
        """Forward pass through the network."""
        X=torch.tensor(X,dtype=torch.float32)
        x = torch.sigmoid(self.fc1(X))  # Apply sigmoid activation
        x = self.fc2(x)  # Pass through the second layer
        self.cache = {'X': X, 'hidden': x}  # Cache for backpropagation
        return x / 3  # Scale the output
    def forward_positions(self,x,motors=0):
        positions = np.zeros(12)
        if type(x)==type(torch.tensor([])): 
            x=x.cpu().detach().numpy().tolist()[0]
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

        if len(x.shape) >= 2:
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

try:
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from torch.distributions import Categorical, Normal

    class CustomNNPolicy(ActorCriticPolicy):
        def __init__(self, observation_space, action_space, lr_schedule, device=None,*args, **kwargs):
            # Initialize the base ActorCriticPolicy
            super(CustomNNPolicy, self).__init__(
                observation_space,
                action_space,
                lr_schedule,
                *args,
                **kwargs,
            )
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #device=torch.device("cpu")
            print(f"Using device: {self.device_}")
            # Define the custom neural network
            self.net = NN(inp=observation_space.shape[0], hidden=64).to(self.device_)
            self.sig = Pattern(0, 0, 0, 0)  # External object, assumed defined elsewhere
            self.val = 0  # Keeps track of the step for position encoding

        def forward(self, obs, *args, **kwargs):
            """Forward pass through policy and value networks."""
            if type(obs)!=type(torch.tensor([])):
                obs=obs.sample()
            obs=torch.tensor(obs,dtype=torch.float32).to(self.device_)
            features = self.net(obs)  # Pass observation through the custom NN
            actions=self.forward_positions(features)
            mean, log_std = self.action_head(action).chunk(2, dim=-1)  # Assuming two outputs: mean and log std
            std = torch.exp(log_std)  # Exponentiate to get the standard deviation
            
            # Create a Normal distribution
            dist = Normal(mean, std)
            
            # Sample an action
            action = dist.sample()
            
            # Calculate log probability of the sampled action
            log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over dimensions of the action space
            
            # Optional: You can store the action or do something with it
            self.val += 1  # Update your custom value if needed
            return actions, features, log_prob
        def forward_positions(self,x,motors=0):
            positions = np.zeros(12)
            x=x.flatten()
            for j, i in enumerate(range(0, 12, 3)):  # Loop through legs assigning phase encoding
                self.sig.set_param(*x[0:4])
                positions[i] = self.sig.forward(x[j] + self.val)
                self.sig.set_param(*x[4:8])
                positions[i + 1] = self.sig.forward(x[j] + self.val)
                self.sig.set_param(*x[8:12])
                positions[i + 2] = self.sig.forward(x[j] + self.val)
            
            positions = motors + ((positions)/5)*50
            positions[positions < 0] = 0
            positions[positions > 50] = 50
            return torch.tensor(positions).to(self.device_)
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
except:
    pass
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
    """n=NN(10,20)
    n.set_genotype(torch.rand(n.genotype.shape))
    motors=[]
    for i in range(300):
        motors.append(n.get_positions(torch.zeros((1,10))).detach().numpy())
    plt.plot(np.array(motors))
    plt.show()"""

    b=Body(10,10)  
    ctrnn = CTRNNQuadruped(imu=1)
    imu_feedback = [0.1,0.3,0.4]  # Example tilt correction
    velocity_feedback = 0.2  # Example velocity correction

    for t in range(100):  # Simulate for 100 steps
        motor_angles = ctrnn.step(imu_feedback, velocity_feedback)
        print(f"Step {t}, Motor Angles: {motor_angles}")  



