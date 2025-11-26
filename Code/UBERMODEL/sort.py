import numpy as np

fitnesses=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/fitness_o.npy")
X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA_o.npy")
print("Starting size",len(fitnesses))
indicies=np.where(fitnesses!=-1000)[0]  
X=X[indicies]
fitnesses=fitnesses[indicies]
average=np.average(fitnesses) 
std=np.std(fitnesses)
indicies=np.where(fitnesses>average+std/2)[0]  
print(len(indicies),len(fitnesses),min(fitnesses),max(fitnesses),np.std(fitnesses))
print("Remainder:",len(indicies),"Reduction to",len(indicies)/len(fitnesses) *100,"%")
X=X[indicies]
np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/models/X_sorted_o",X)