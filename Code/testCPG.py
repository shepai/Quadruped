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
        self.b=-4
    def sigma(self,x):
        return 1/(1+np.exp(-x))
    def forward(self,I=0):
        #TA=-self.A.T + np.dot(self.act(self.A.T),self.weights)+ I.T +self.bias #forward pass
        O=self.sigma(self.A+self.b)
        self.A = self.A + (self.dt/self.Tau)*(-self.A + np.dot(self.weights,O)+I)
        return self.A
    
Neurons=6
model=Network(Neurons)  
arc=[deepcopy(model),deepcopy(model),deepcopy(model),deepcopy(model)]

SAMPLE=300
data=np.zeros((4,SAMPLE,Neurons))
last_Input=np.zeros_like(arc[0].A)
for i in range(SAMPLE):
    for j in range(4):
        last_Input=np.random.random(Neurons)*10
        nodes=arc[j].forward(I=last_Input)
        lastInput=nodes.copy()
        data[j][i]=nodes.flatten()

data=np.array(data)
print(data.shape)
for j in range(Neurons):
    #for i in range(len(data[j].T)):
    #plt.plot(data[j].T[4],label="Output network"+str(j))
    plt.plot(data[0].T[j],label="Output neuron"+str(j))
    
plt.legend(loc="upper left")
plt.show()
