if __name__=="__main__":
    import sys
    sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')#
    sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
datapath="/its/home/drs25/Documents/GitHub/Quadruped/"
datapath="C:/Users/dexte/Documents/GitHub/Quadruped/"
from environment import *
from CPG import *
import time
from copy import deepcopy
import pickle

env=environment(0)
def fitness_(robot,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        distances=np.array(history['positions'])-np.array([robot.start])#euclidean_distance(np.array(history['positions']),np.array([robot.start]))
        distancesX=distances[-1][0]
        distancesY=distances[-1][1]
        distancesZ=distances[-1][2]
        fitness+=distancesX - (distancesY+distancesZ)/10 #np.sum(distances)
    if robot.hasFallen(): fitness=0
    if type(fitness)!=type(0): 
        try:
            if type(fitness)==type([]): fitness=float(fitness[0])
            else:fitness=float(fitness)
        except:
            print("shit",fitness,np.array(history['motors']).shape,np.array(history['positions']).shape,np.array(history['orientations']).shape)
            fitness=0
    if fitness<0: fitness=0
    return fitness
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2,axis=1))

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2,axis=1))
#agent goes in population generation
#initial
population_size=50
#g=geno_gen(6,population_size)
population=[CTRNNQuadruped() for _ in range(population_size)]#np.random.choice([50, 20, 0,0,0,0,-20],(150,15,12)) #12 motors, 15 steps
"""population=[]
for i in range(population_size):
    b=Body(6,4)
    b.set_genotype(g.population[i])
    population.append(deepcopy(b))"""
    
fitnesses=np.zeros((population_size,))
generations=800
t_start=time.time()
#get fitnesses
for i in range(len(fitnesses)):
    fitnesses[i],_=env.runTrial(population[i],100,delay=0,fitness=fitness_)
    print(i,"/",len(fitnesses), fitnesses[i])

for gen in range(generations):
    clear = lambda: os.system('clear')
    print("Generation ",gen+1,"Best fitness",np.max(fitnesses))
    ind1=np.random.randint(0,len(fitnesses)-1)
    ind2=np.random.randint(0,len(fitnesses)-1)
    geno=deepcopy(population[ind1])
    f,_=env.runTrial(geno,100,delay=False,fitness=fitness_) #run trial
    if f>fitnesses[ind2]: #selection
        mutated=deepcopy(geno)
        mutated.mutate()
        fitnesses[ind2],motors=env.runTrial(mutated,100,delay=False,fitness=fitness_)
        population[ind2]=deepcopy(mutated)
    elif fitnesses[ind2]>f:
        mutated=deepcopy(geno)
        mutated.mutate()
        fitnesses[ind1],motors=env.runTrial(mutated,100,delay=False,fitness=fitness_)
        population[ind1]=deepcopy(mutated)
    #if gen%10==0:
        #runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],150)
#play the trials on reapeat
    if gen%10==0:
        with open(datapath+'/models/genotypes_5.pkl', 'wb') as f:
            pickle.dump(population, f)
        np.save(datapath+"/models/fitnesses_5",fitnesses)


env.runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],150,fitness=fitness_)
print("top fitness:",population[np.where(fitnesses==np.max(fitnesses))[0][0]])
p.disconnect()


t_passed=time.time()-t_start
print("********************************\n\n\n\nTIME IT TOOK:",t_passed/(60*60),"Hours")