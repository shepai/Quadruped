import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import sys
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
from PIL import Image
from environment import *
from CPG import *
from CPG import *
import copy
datapath="/its/home/drs25/Documents/GitHub/Quadruped/"
#datapath="C:/Users/dexte/Documents/GitHub/Quadruped/"
np.set_printoptions(suppress=True, precision=4)
from GA import *
def load_data(fitness="F1"):
    best_CPGs=[]
    fitnesses=[]
    history=[]
    trials=40
    dt=0.1
    for _ in range(1,trials):
        filename = datapath+'/models/genotypes_dt'+str(dt)+"_6_neurons_"+str(_)+"_"+fitness
        population=None
        with open(filename+'.pkl', 'rb') as f:
            population = pickle.load(f)
        f_=np.load((filename+'.npy').replace("genotypes","fitnesses"))
        history.append(np.load((filename+'.npy').replace("genotypes","history")))
        best_geno=np.argmax(f_)
        best_CPGs.append(population[best_geno])
        fitnesses.append(np.max(f_))

    fitnesses=np.array(fitnesses).reshape(trials-1)
    history=np.array(history)[:,:-10]
    return fitnesses,history,best_CPGs

class newGait(CTRNNQuadruped):
    def mutate(self,rate=0.2):
        probailities=np.random.random(self.geno.shape)
        self.geno[np.where(probailities<rate)]+=np.random.normal(0,4,self.geno[np.where(probailities<rate)].shape)
        self.set_genotype(self.geno)

def RUN(dt=0.1,sho=0,trial=0,generations=300,fit=fitness_,fric=0.5):
    env=environment(sho,friction=fric)
    #agent goes in population generation
    #initial
    fitnesses,history,best_CPGs=load_data(fitness="F1")

    best_walk=best_CPGs[np.argmax(fitnesses)]
    population_size=50
    #g=geno_gen(6,population_size)
    population=[]
    for i in range(population_size):
        cpg=newGait()
        cpg.set_genotype(best_walk.geno)
        cpg.mutate(rate=0.5)
        population.append(copy.deepcopy(cpg))
        
    fitnesses=np.zeros((population_size,))
    
    t_start=time.time()
    #get fitnesses
    for i in range(len(fitnesses)):
        fitnesses[i],_,_2=env.runTrial(population[i],100,delay=0,fitness=fit)
        #print(i,"/",len(fitnesses), fitnesses[i])
    history=np.zeros((generations,))
    for gen in range(generations):
        
        if gen%50==0:
            print("Generation ",gen+1,"Best fitness",np.max(fitnesses))
        ind1=np.random.randint(0,len(fitnesses)-1)
        ind2=np.random.randint(0,len(fitnesses)-1)
        if fitnesses[ind1]>fitnesses[ind2]: #selection
            geno=deepcopy(population[ind1])
            mutated=deepcopy(geno)
            mutated.mutate()
            fitnesses[ind2],motors,_=env.runTrial(mutated,100,delay=False,fitness=fit)
            population[ind2]=deepcopy(mutated)
        elif fitnesses[ind2]>fitnesses[ind1]:
            geno=deepcopy(population[ind2])
            mutated=deepcopy(geno)
            mutated.mutate()
            fitnesses[ind1],motors,_=env.runTrial(mutated,100,delay=False,fitness=fit)
            population[ind1]=deepcopy(mutated)
        history[gen]=np.max(fitnesses)
    #play the trials on reapeat
        if gen%10==0:
            with open(datapath+'/models/genotypes_dt'+str(dt)+"_"+str(trial)+str(fric)+'.pkl', 'wb') as f:
                pickle.dump(population, f)
            np.save(datapath+'/models/fitnesses_dt'+str(dt)+"_"+str(trial)+str(fric),fitnesses)
            np.save(datapath+'/models/history_dt'+str(dt)+"_"+str(trial)+str(fric),history)


    env.runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],150,fitness=fit)
    print("top fitness:",np.max(fitnesses))
    p.disconnect()


    t_passed=time.time()-t_start
    print("********************************\n\n\n\nTIME IT TOOK:",t_passed/(60*60),"Hours")

if __name__=="__main__":
    c=0
    for i in range(1,40):
        for j in np.arange(0,1,0.05):
            os.system('cls' if os.name == 'nt' else 'clear')
            calc=len(range(0,40))*len( np.arange(0,1,0.05))
            print(i,j,c/calc *100,"%")
            RUN(dt=0.1,sho=0,trial="already_evolved"+str(i),fit=F1,fric=j)
            c+=1
