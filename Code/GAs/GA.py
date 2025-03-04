if __name__=="__main__":
    import sys
    sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')#
    sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
from environment import *
from CPG import *
import time
from copy import deepcopy
import pickle

env=environment(1)
def fitness_(robot,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        #calculate the phase length of the hip
        oscillations = np.diff(np.array(history['motors'])[:,3])
        zero_crossings = np.where(np.diff(np.sign(oscillations)) != 0)[0] + 1  # +1 to correct index shift
        diff=np.diff(zero_crossings)
        if type(np.diff(zero_crossings))==type([]) or type(np.diff(zero_crossings))==type(np.array([])): 
            if len(np.diff(zero_crossings))>0:diff=np.average(diff)
            else: diff=0
        fitness+=diff/10 #more phase is betters
        #distance over time
        distances=euclidean_distance(np.array(history['positions']),np.array([robot.start]))
        distances=np.diff(distances)

        fitness+=np.sum(distances)
        #orientationo over time#
        stability_penalty = np.mean(np.linalg.norm(np.array(history['orientations']) - np.array(robot.start_orientation), axis=1))
        jerkiness_penalty = np.sum(np.linalg.norm(np.diff(np.array(history['orientations']), axis=0), axis=1))
        fitness -= 0.01 * stability_penalty + 0.001 * jerkiness_penalty
    else: #basic fitness
        distance = euclidean_distance(np.array([robot.start]),np.array([robot.getPos()]))
        orientation_penalty = np.linalg.norm(np.array(robot.getOrientation()) - np.array(robot.start_orientation)) 
        distance *= np.exp(-0.1 * orientation_penalty)  # Penalize unstable rotations
        direction_vector = np.array(robot.getPos()[0:2]) - np.array(robot.start[0:2])
        goal_direction = np.array([1, 0])  # Example: moving in +x direction
        direction_reward = np.dot(direction_vector, goal_direction) / (np.linalg.norm(direction_vector) + 1e-6)
        distance *= (1 + direction_reward)
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
generations=3000
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
        with open('/its/home/drs25/Documents/GitHub/Quadruped/models/genotypes_5.pkl', 'wb') as f:
            pickle.dump(population, f)
        np.save("/its/home/drs25/Documents/GitHub/Quadruped/models/fitnesses_5",fitnesses)


env.runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],150,fitness=fitness_)

p.disconnect()


t_passed=time.time()-t_start
print("********************************\n\n\n\nTIME IT TOOK:",t_passed/(60*60),"Hours")