import numpy as np
from copy import deepcopy
from agent import *
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
        self.geno=np.array([a,h,b,k])
    def forward(self,t):
        g=self.geno
        return g[0] * np.sin((t-g[1])/g[2]) +g[3]
    def set_param(self,a,h,b,k):
        self.geno=np.array([a,h,b,k])

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
        positions=np.array(positions)/1.5
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

class NN:
    def __init__(self,inp,hidden):
        out=16
        self.fc1=np.random.random((inp,hidden))
        self.b1=np.random.random((1,hidden))
        self.fc2=np.random.random((hidden,out))
        self.b2=np.random.random((1,out))
        self.genotype=np.concatenate([self.fc1.flatten(),self.b1.flatten(),self.fc2.flatten(),self.b2.flatten()])
        self.sig=Pattern(0,0,0,0)
        self.val=0
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def forward(self,X):
        x=self.sigmoid(np.dot(X,self.fc1))+self.b1
        x=np.dot(x,self.fc2)+self.b2
        return x
    def get_positions(self,x):
        x=self.forward(x)[0]
        positions=np.zeros(12,)
        for j,i in enumerate(range(0,12,3)): #loop through legs assigning phase encoding
            self.sig.set_param(*x[0:4])
            positions[i]=self.sig.forward(x[j]+self.val)
            self.sig.set_param(*x[4:8])
            positions[i+1]=self.sig.forward(x[j]+self.val)
            self.sig.set_param(*x[8:12])
            positions[i+2]=self.sig.forward(x[j]+self.val)
        self.val+=1
        positions[positions<0]=0
        positions[positions>30]=30
        return positions
    def set_genotype(self,geno):
        self.genotype=geno.copy()
        self.genotype[self.genotype>5]=5
        self.genotype[self.genotype<-5]=-5
        self.fc1=self.genotype[0:len(self.fc1.flatten())].reshape(self.fc1.shape)
        idx=len(self.fc1.flatten())
        self.b1=self.genotype[idx:idx+len(self.b1.flatten())].reshape(self.b1.shape)
        idx+=len(self.b1.flatten())
        self.fc2=self.genotype[idx:idx+len(self.fc2.flatten())].reshape(self.fc2.shape)
        idx+=len(self.fc2.flatten())
        self.b2=self.genotype[idx:idx+len(self.b2.flatten())].reshape(self.b2.shape)
    def mutate(self):
        self.set_genotype(self.genotype+np.random.normal(0,2,self.genotype.shape))

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
    n.set_genotype(np.random.random(n.genotype.shape))
    motors=[]
    for i in range(300):
        motors.append(n.get_positions(np.zeros((1,10))))
    plt.plot(np.array(motors))
    plt.show()
    



