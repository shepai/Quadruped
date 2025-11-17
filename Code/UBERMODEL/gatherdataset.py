import numpy as np 
import os 
import pickle
import sys
import re
import time 
from copy import deepcopy
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, '/its/home/drs25/Quadruped/Code')
sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, '/its/home/drs25/Quadruped/Code/GAs')
from environment import *
from CPG import *
# look through a folder directory and gather all files within it
def list_all_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.join(root, f))
    return files

# select out the ones with genotype in the name 
files = list_all_files("/its/home/drs25/Quadruped/models/")
print("FILES",len(files))
genotypes=[]
for i in range(len(files)):
    if "genotypes" in files[i]:
        genotypes.append(files[i])
print("GENOTYPES",len(genotypes))
checked=[]
fitness=[]
history=[]
friction=[]
estimated_time=[]
#check it is a CTRNN for quadruped 
for i in range(len(genotypes)): #len(genotypes)
    print(i,"/",len(genotypes))
    if "friction" in genotypes[i]:
        friction_value=0
        match = re.search(r"friction([0-9]*\.?[0-9]+)", genotypes[i])
        if match:
            friction_value = float(match.group(1))
            with open(genotypes[i], 'rb') as f:
                population = pickle.load(f)
            for j in range(len(population)): #len(population)
                if isinstance(population[j], CTRNNQuadruped): #run the genotype and gather fitness motor commands
                    try:
                        t1=time.time()
                        checked.append(deepcopy(population[j]))
                        env=environment(0,friction=friction_value)
                        _fit,hist,_=env.runTrial(population[j],50)
                        env.close()
                        history.append(deepcopy(hist))
                        friction.append(friction_value)
                        fitness.append(_fit)
                        t2=time.time()
                        estimated_time.append(t2-t1)
                    except:
                        print("An older one... that does not check out")
                        pass 
        average_time=sum(estimated_time)/len(estimated_time)
        time_left=average_time*(len(genotypes)-i)
        print(i,"/",len(genotypes),"Time left:",time_left/60/60,"hours")
print("Acceptable genotypes:",len(checked))

#use positions to estimate trajectory vectors, dataset should be T x vector, friction, motor positions in, feet reading then motor position out
X=[] #store 
y=[]
for i in range(len(checked)):
    f=friction[i] 
    X_T=[]
    y_T=[]
    if len(history[i]['positions'])-1 == 499:
        for t in range(len(history[i]['positions'])-1): #store into a single vector over time
            v1 = np.array(history[i]['positions'][t])
            v2 = np.array(history[i]['positions'][t+1])
            vector = v2 - v1
            motors_current=np.array(history[i]['motors'][t])
            motors_next=np.array(history[i]['motors'][t+1])
            feet=history[i]['feet'][t]
            overall_vector=np.concatenate([motors_current,vector,feet,friction])
            X_T.append(overall_vector)
            y_T.append(motors_next)
        X.append(np.array(X_T))
        y.append(np.array(y_T))
X=np.array(X)
y=np.array(y)
print("X size:",X.shape)
print("y size:",y.shape)

np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA",X)
np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/y_DATA",y)
#save file 