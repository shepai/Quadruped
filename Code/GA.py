from environment import *
from CPG import *

env=environment(True)
def fitness_(robot):
    distance = (euclidean_distance(robot.getPos()[0:2],robot.start[0:2])*0.8 - euclidean_distance(robot.start_orientation,robot.getOrientation())*0.1)-robot.get_self_collision_count()*0.01
    if distance<0: distance=0
    if robot.hasFallen(): return 0
    return distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
#initial
population_size=10
population=[CPG(6) for _ in range(population_size)]#np.random.choice([50, 20, 0,0,0,0,-20],(150,15,12)) #12 motors, 15 steps
fitnesses=np.zeros((population_size,))
generations=1000

#get fitnesses
for i in range(len(fitnesses)):
    fitnesses[i]=env.runTrial(population[i],200,delay=0,fitness=fitness_)
    print(i,"/",len(fitnesses), fitnesses[i])

for gen in range(generations):
    clear = lambda: os.system('clear')
    print("Generation ",gen+1,"Best fitness",np.max(fitnesses))
    ind1=np.random.randint(0,len(fitnesses)-1)
    ind2=np.random.randint(0,len(fitnesses)-1)
    geno=population[ind1]
    f=env.runTrial(geno,100,delay=False,fitness=fitness_) #run trial
    if f>fitnesses[ind2]: #selection
        mutated=deepcopy(geno)
        mutated.mutate()
        fitnesses[ind2]=env.runTrial(mutated,200,delay=False,fitness=fitness_)
        population[ind2]=mutated
    else:
        mutated=deepcopy(geno)
        mutated.mutate()
        fitnesses[ind1]=env.runTrial(mutated,200,delay=False,fitness=fitness_)
        population[ind1]=mutated
    #if gen%10==0:
        #runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],150)
#play the trials on reapeat
import pickle
with open('genotypes.pkl', 'wb') as f:
    pickle.dump(population, f)
np.save("fitnesses",fitnesses)

p.disconnect()
p.connect(p.GUI)
env.runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],150)
p.disconnect()