if __name__=="__main__":
    import sys
    sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
    #sys.path.insert(1, '/its/home/drs25/Quadruped/Code')
    sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
datapath="/its/home/drs25/Documents/GitHub/Quadruped/"
#datapath="C:/Users/dexte/Documents/GitHub/Quadruped/"
#datapath="/its/home/drs25/Quadruped/"

from environment import *
from CPG import CTRNNQuadruped
import pickle
from copy import deepcopy
from GA_gaitchange import RUN_hillclimber, F3
with open(datapath+'/models/6_neurons/genotypes_dt0.1__6_neurons_0_F10.5.pkl', 'rb') as f:
    population = pickle.load(f)
fitnesses=np.load(datapath+"/models/6_neurons/fitnesses_dt0.1__6_neurons_0_F10.5.npy")
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
env=environment(True,1,datapath+"/assets/videos/lowspeed.mp4",UI=0)
#low
env.INCREASE=-30
env.BALANCE=30
#high
env.INCREASE=10
env.BALANCE=-10
#normal
env.INCREASE=0
env.BALANCE=0
env.STRIDE=1
#population[index].dt=0.01
for fric in np.arange(0,1,0.05):
    #firstly generate the perfect gait
    print("PRGRESS",fric,stride,body,speed)
    best_geno=RUN_hillclimber(dt=0.1,sho=0,trial="frictions_and_stuff/hillclimber_"+str(i)+"_friction",fit=F3,fric=fric,encoding=[stride,body,speed])
    for stride in range(3):
            for body in range(3):
                for speed in range(3):
                    copy=deepcopy(best_geno)
                    env=environment(0,300,friction=fric)
                    if stride==0:#low
                        env.STRIDE=0.5
                    elif stride==1:#high
                        env.STRIDE=2
                    else:#normal
                        env.STRIDE=1
                    if body==0:#low
                        env.BALANCE=30
                        env.INCREASE=-30
                    elif body==1:#high
                        env.BALANCE=-10
                        env.INCREASE=0
                    else:#normal
                        env.BALANCE=0
                    if speed==0:
                        copy.tau*=0.001
                    elif speed==1:
                        copy.tau*=10000
                    #then see how the changes impact
                    fit,history,photos=env.runTrial(copy,50,delay=0,fitness=fitness_,photos=-1)
                    env.stop()

                    np.savez(datapath+"/models/frictions_and_stuff/"+str(fric)+"_"+str(stride)+"_"+str(body)+"_"+str(speed),history,allow_pickle=True)

if len(photos)>1:
    print("saving photos")
    np.save(datapath+"/assets/lowspeed",np.array(photos))