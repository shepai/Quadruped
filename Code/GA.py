path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
#path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
import Quadruped
import numpy as np

# Initialize the PyBullet physics engine
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
def fitness(robot):
    x,y,z=robot.getPos()
    sx,sy,sz=robot.start
    distance = maths.sqrt((x - sx)**2 + (y - sy)**2)
    return distance

def runTrial(agent,generations,delay=True):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF('plane.urdf')
    initial_position = [0, 0, 0.2]  # x=1, y=2, z=0.5
    initial_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
    flags = p.URDF_USE_SELF_COLLISION
    robot_id = p.loadURDF(path+"Quadruped.urdf", initial_position, initial_orientation,flags=flags)
    for i in range(100):
        p.stepSimulation()
        p.setTimeStep(1./24.)
    quad=Quadruped.Quadruped(p,robot_id)
    quad.reset()
    #GA
    for i in range(generations):
        for j in range(agent.shape[0]):
            positions=agent[j]
            quad.setPositions(positions)
            for k in range(10):
                p.stepSimulation()
                if delay: time.sleep(1./240.)
                else: p.setTimeStep(1./240.)
    f=fitness(quad)
    del quad
    p.removeBody(robot_id)
    return f

def mutate(array,probability=0.2):
    flip_mask = np.random.rand(*array.shape) < probability
    array[flip_mask] = np.where(array[flip_mask] == 30, 0, 30)
    return array

#initial
population=np.random.choice([30, 0],(150,15,12)) #12 motors, 15 steps
fitnesses=np.zeros((150,))
generations=2000

#get fitnesses
for i in range(len(fitnesses)):
    fitnesses[i]=runTrial(population[i],100,delay=False)
    print(i,"/",len(fitnesses), fitnesses[i])

for gen in range(generations):
    print("Generation ",gen+1)
    ind1=np.random.randint(0,len(fitnesses)-1)
    ind2=np.random.randint(0,len(fitnesses)-1)
    geno=population[ind1].copy()
    f=runTrial(geno,100,delay=False) #run trial
    if f>fitnesses[ind2]: #selection
        mutated=mutate(geno)
        fitnesses[ind2]=runTrial(mutated,100,delay=False)
        population[ind2]=mutated.copy()
    else:
        mutated=mutate(geno)
        fitnesses[ind1]=runTrial(mutated,100,delay=False)
        population[ind1]=mutated.copy()
    #if gen%10==0:
        #runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],150)
#play the trials on reapeat
while True:
    for u in range(len(fitnesses)):
        runTrial(population[u],100)

p.disconnect()