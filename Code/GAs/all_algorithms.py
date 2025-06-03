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

def RUN_diff(dt=0.1,sho=0,trial=0,generations=300,fit=fitness_,fric=0.5):
    env=environment(sho,friction=fric)
    env.dt=dt
    population_size=50
    mutation_rate=0.2
    ga=Differential(population_size, generations, mutation_rate,0.2)
    ga.initialize_population(CTRNNQuadruped, [4,3,0.1,0])
    history,fitnesses=ga.evolve(env, fit, 20,outputs=1)
    t_start=time.time()
    #get fitnesses
    with open(datapath+'/models/genotypes_dt'+str(dt)+"_"+str(trial)+str(fric)+'.pkl', 'wb') as f:
        pickle.dump(ga.pop, f)
    np.save(datapath+'/models/fitnesses_dt'+str(dt)+"_"+str(trial)+str(fric),fitnesses)
    np.save(datapath+'/models/history_dt'+str(dt)+"_"+str(trial)+str(fric),history)

    env.runTrial(ga.pop[np.where(fitnesses==np.max(fitnesses))[0][0]],150,fitness=fit)
    print("top fitness:",np.max(fitnesses))
    p.disconnect()

    t_passed=time.time()-t_start
    print("********************************\n\n\n\nTIME IT TOOK:",t_passed/(60*60),"Hours")


def RUN_microbial(dt=0.1,sho=0,trial=0,generations=300,fit=fitness_,fric=0.5):
    env=environment(sho,friction=fric)
    env.dt=dt
    population_size=50
    mutation_rate=0.2
    ga=Microbial_GA(population_size, generations, 0.2)
    ga.initialize_population(CTRNNQuadruped, [4,3,0.1,0])
    history,fitnesses=ga.evolve(env, fit, 20,outputs=1)
    t_start=time.time()
    #get fitnesses
    with open(datapath+'/models/genotypes_dt'+str(dt)+"_"+str(trial)+str(fric)+'.pkl', 'wb') as f:
        pickle.dump(ga.pop, f)
    np.save(datapath+'/models/fitnesses_dt'+str(dt)+"_"+str(trial)+str(fric),fitnesses)
    np.save(datapath+'/models/history_dt'+str(dt)+"_"+str(trial)+str(fric),history)

    env.runTrial(ga.pop[np.where(fitnesses==np.max(fitnesses))[0][0]],150,fitness=fit)
    print("top fitness:",np.max(fitnesses))
    p.disconnect()

    t_passed=time.time()-t_start
    print("********************************\n\n\n\nTIME IT TOOK:",t_passed/(60*60),"Hours")

def RUN_hillclimber(dt=0.1,sho=0,trial=0,generations=300,fit=fitness_,fric=0.5):
    env=environment(sho,generations,friction=fric)
    env.dt=dt
    population_size=50
    mutation_rate=0.2
    ga=Hillclimbers(population_size, generations, 0.2)
    ga.initialize_population(CTRNNQuadruped, [4,3,0.1,0])
    history,fitnesses=ga.evolve(env, fit, 20,outputs=1)
    t_start=time.time()
    #get fitnesses
    with open(datapath+'/models/genotypes_dt'+str(dt)+"_"+str(trial)+str(fric)+'.pkl', 'wb') as f:
        pickle.dump(ga.pop, f)
    np.save(datapath+'/models/fitnesses_dt'+str(dt)+"_"+str(trial)+str(fric),fitnesses)
    np.save(datapath+'/models/history_dt'+str(dt)+"_"+str(trial)+str(fric),history)

    env.runTrial(ga.pop[np.where(fitnesses==np.max(fitnesses))[0][0]],150,fitness=fit)
    print("top fitness:",np.max(fitnesses))
    p.disconnect()

    t_passed=time.time()-t_start
    print("********************************\n\n\n\nTIME IT TOOK:",t_passed/(60*60),"Hours")
if __name__=="__main__":
    for i in range(20):
        #RUN_microbial(dt=0.1,sho=0,generations=300,trial="LongMicrobial_"+str(i)+"_friction",fit=F3,fric=0.5)
        RUN_hillclimber(dt=0.1,sho=0,generations=300,trial="Hillclimber_F1_"+str(i)+"_friction",fit=F1,fric=0.5)
        RUN_hillclimber(dt=0.1,sho=0,generations=300,trial="Hillclimber_F2_"+str(i)+"_friction",fit=F2,fric=0.5)
        RUN_hillclimber(dt=0.1,sho=0,generations=300,trial="Hillclimber_F3_"+str(i)+"_friction",fit=F3,fric=0.5)
        #RUN_diff(dt=0.1,sho=0,generations=300,trial="Differential_"+str(i)+"_friction",fit=F3,fric=0.5)

    """c=0
    for i in range(0,40):
        for j in np.arange(0,1,0.05):
            os.system('cls' if os.name == 'nt' else 'clear')
            calc=len(range(0,40))*len( np.arange(0,1,0.05))
            print(i,j,c/calc *100,"%")
            RUN_hillclimber(dt=0.1,sho=0,trial="hill_climber_"+str(i)+"_friction",fit=F3,fric=j)
            c+=1"""
