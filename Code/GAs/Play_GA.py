if __name__=="__main__":
    import sys
    sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
    sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
datapath="/its/home/drs25/Documents/GitHub/Quadruped/"
datapath="C:/Users/dexte/Documents/GitHub/Quadruped/"
from environment import *
from CPG import *
import pickle
with open(datapath+'/models/genotypes_dt0.1_6_neurons_0.pkl', 'rb') as f:
    population = pickle.load(f)
fitnesses=np.load(datapath+"/models/fitnesses_dt0.1_6_neurons_0.npy")
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

#find best fitness
best=0
index=0

index=np.argmax(fitnesses)
env=environment(True,1,datapath+"/assets/videos/example_dt0.2_6_neurons_0.mp4")
population[index].dt=1
fit,mot,photos=env.runTrial(population[index],50,delay=0,fitness=fitness_,photos=10)
env.stop()
np.savez(datapath+"/Code/GAs/motors_dt0.2_6_neurons_0",mot,allow_pickle=True)

if len(photos)>1:
    print("saving photos")
    np.save(datapath+"/assets/frames",np.array(photos))