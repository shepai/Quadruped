if __name__=="__main__":
    import sys
    sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')#
    sys.path.insert(1, '/its/home/drs25/Quadruped/Code')#
    sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
datapath="/its/home/drs25/Quadruped/"
#datapath="C:/Users/dexte/Documents/GitHub/Quadruped/"
from environment import *
from CPG import *
import time
from copy import deepcopy
import pickle
import os
from fitnesses import *
from genetic_algorithms import *


def RUN_hillclimber(dt=0.1,sho=0,trial=0,generations=300,fit=fitness_,fric=0.5,encoding=[0,0,0],dir_=""):
    env=environment(sho,generations,friction=fric)
    if encoding[0]==-1:#low
        env.STRIDE=0.5
    elif encoding[0]==1:#high
        env.STRIDE=2
    else:#normal
        env.STRIDE=1
    if encoding[1]==-1:#low
        env.BALANCE=30
        env.INCREASE=-30
    elif encoding[1]==1:#high
        env.BALANCE=-10
        env.INCREASE=0
    else:#normal
        env.BALANCE=0
    env.dt=dt
    population_size=50
    mutation_rate=0.2
    ga=Hillclimbers(population_size, generations, 0.2)
    ga.initialize_population(CTRNNQuadruped, [4,3,0.1,0])
    for i in range(len(ga.pop)):
        if encoding[2]==-1:
            ga.pop[i].tau*=0.001
        elif encoding[2]==1:
            ga.pop[i].tau*=10000

    history,fitnesses=ga.evolve(env, fit, 20,outputs=1)
    t_start=time.time()
    #get fitnesses
    name=str(encoding[0])+"-"+str(encoding[1])+"-"+str(encoding[2])
    with open(datapath+dir_+'genotypes_dt'+str(dt)+"_"+str(trial)+str(fric)+"_"+name+'.pkl', 'wb') as f:
        pickle.dump(ga.pop, f)
    np.save(datapath+dir_+'fitnesses_dt'+str(dt)+"_"+str(trial)+str(fric)+"_"+name,fitnesses)
    np.save(datapath+dir_+'history_dt'+str(dt)+"_"+str(trial)+str(fric)+"_"+name,history)

    env.runTrial(ga.pop[np.where(fitnesses==np.max(fitnesses))[0][0]],150,fitness=fit)
    print("top fitness:",np.max(fitnesses))
    p.disconnect()

    t_passed=time.time()-t_start
    print("********************************\n\n\n\nTIME IT TOOK:",t_passed/(60*60),"Hours")
    return ga.pop[np.argmax(fitnesses)] #return best
if __name__=="__main__":
    c=0
    for stride in range(3):
        for body in range(3):
            for speed in range(3):
                for i in range(0,40):
                    for j in np.arange(0,1,0.05):
                        os.system('cls' if os.name == 'nt' else 'clear')
                        calc=len(range(0,40))*len( np.arange(0,1,0.05))
                        print(i,j,c/calc *100,"%")
                        RUN_hillclimber(dt=0.1,sho=0,trial="/models/frictionGaitSelect/hill_climber_"+str(i)+"_friction",fit=F3,fric=j,encoding=[stride,body,speed])
                        c+=1
                        progress=[stride,body,speed,i,j]
                        file=open("/its/home/drs25/Quadruped/Code/GAs/track_progress.txt","w")
                        file.write(str(progress))
                        file.close()
