import numpy as np
from copy import deepcopy
from agent import *

class Network:
    def __init__(self,num,std=5):
        self.A=np.zeros((num))
        self.weights=np.random.normal(0,std,(num,num))
        np.fill_diagonal(self.weights, 0) #zero self weights
        self.Tau=1
        self.dt=0.01
        self.b=0
    def sigma(self,x):
        return 1/(1+np.exp(-x))
    def forward(self,I=0):
        #TA=-self.A.T + np.dot(self.act(self.A.T),self.weights)+ I.T +self.bias #forward pass
        O=self.sigma(self.A+self.b)
        self.A = self.A + (self.dt/self.Tau)*(-self.A + np.dot(self.weights,O)+I)
        return self.A
    
class CPG(agent):
    def __init__(self,num_neurons):
        agent.__init__(self)
        self.cpg=Network(num_neurons)
        self.num_neurons=num_neurons
        self.geno=np.concatenate([self.cpg.weights.flatten(),[self.cpg.b],[self.cpg.Tau]])
        self.set_genotype(self.geno)
        self.populateBody()
    def populateBody(self):
        self.body=[]
        for i in range(4): #one for each leg
            self.body.append(deepcopy(self.cpg))
    def get_positions(self,inputs):
        positions=[]
        Inputs=np.zeros((self.num_neurons))
        Inputs[0]+=inputs[0] #add proprioception
        Inputs[1]+=inputs[1]
        Inputs[2]+=inputs[2]
        for cpg in self.body:
            out=cpg.forward(I=Inputs) #forward
            Inputs=out
            Inputs[0]+=inputs[0]
            Inputs[1]+=inputs[1]
            Inputs[2]+=inputs[2]
            #Inputs=self.cpg.sigma(Inputs)*15
            Inputs[Inputs<-40]=-40
            Inputs[Inputs>40]=40
            positions.append(Inputs[0]) #select neuron outputs as proportional to motor
            positions.append(Inputs[1])
            positions.append(Inputs[2])
        return np.array(positions)
    def set_genotype(self, values):
        self.cpg.weights=values[0:len(self.cpg.weights.flatten())].reshape(self.cpg.weights.shape)
        self.cpg.b=values[len(self.cpg.weights.flatten())]
        self.cpg.b = -4 if self.cpg.b < -4 else self.cpg.b #cap bias
        self.cpg.b = 4 if self.cpg.b > 4 else self.cpg.b #cap bias
        self.cpg.Tau=values[len(self.cpg.weights.flatten())+1]
        self.cpg.weights[self.cpg.weights>16]=16 #cap weights
        self.cpg.weights[self.cpg.weights<-16]=-16 #cap weights
        self.populateBody()
        return super().set_genotype(np.concatenate([self.cpg.weights.flatten(),[self.cpg.b],[self.cpg.Tau]]))
    

if __name__ == "__main__":
    from environment import *
    env=environment(True)
    agent=CPG(6)
    env.runTrial(agent,1000,True)

