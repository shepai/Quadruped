import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from copy import deepcopy

matplotlib.use('TkAgg')

class Network:
    def __init__(self,num):
        self.A=np.zeros((num))
        self.weights=np.random.normal(0,2,(num,num))
        np.fill_diagonal(self.weights, 0) #zero self weights
        self.Tau=1
        self.dt=0.01
        self.b=0
    def act(self,x): #activation function
        return (1 + np.exp(-(self.b+x)))**-1
    def sigma(self,x):
        return 1/(1+np.exp(-x))
    def forward(self,I=0):
        #TA=-self.A.T + np.dot(self.act(self.A.T),self.weights)+ I.T +self.bias #forward pass
        O_j=self.act(self.A)
        total_inputs=np.dot(self.weights, O_j)
        self.A += (self.dt * (1/ self.Tau))
        return self.A

model=Network(6)  
arc=[deepcopy(model),deepcopy(model),deepcopy(model),deepcopy(model)]

SAMPLE=300
data=np.zeros((4,SAMPLE,6))
last_Input=np.zeros_like(arc[0].A)
for i in range(100):
    for j in range(4):
        nodes=arc[j].forward(I=last_Input)
        lastInput=nodes.copy()
        data[j][i]=nodes.flatten()

data=np.array(data)
for j in range(4):
    #for i in range(len(data[j].T)):
    #plt.plot(data[j].T[4],label="Output network"+str(j))
    plt.plot(data[0].T[j],label="Output neuron"+str(j))
    
plt.legend(loc="upper left")
plt.show()
