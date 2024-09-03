path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
import Quadruped
import numpy as np

# Initialize the PyBullet physics engine
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def fitness(robot):
    distance = (euclidean_distance(robot.getPos()[0:2],robot.start[0:2])*0.8 - euclidean_distance(robot.start_orientation,robot.getOrientation())*0.1)-robot.get_self_collision_count()*0.01
    if distance<0: distance=0
    return distance

def runTrial(agent,generations,delay=True):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF('plane.urdf')
    initial_position = [0, 0, 0.3]  # x=1, y=2, z=0.5
    initial_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
    flags = p.URDF_USE_SELF_COLLISION
    robot_id = p.loadURDF(path+"Quadruped_prestip.urdf", initial_position, initial_orientation,flags=flags)
    for i in range(100):
        p.stepSimulation()
        p.setTimeStep(1./24.)
    quad=Quadruped.Quadruped(p,robot_id)
    quad.neutral=[-10,0,30,0,0,0,0,0,0,0,0,0]
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
    array[flip_mask] = np.random.choice([50,20,0,0,-20], size=np.count_nonzero(flip_mask))
    return array

#initial
population=mutate(np.zeros((150,15,12)),probability=0.4)#np.random.choice([50, 20, 0,0,0,0,-20],(150,15,12)) #12 motors, 15 steps
fitnesses=np.zeros((150,))
generations=2000

#get fitnesses
for i in range(len(fitnesses)):
    fitnesses[i]=runTrial(population[i],100,delay=0)
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
np.save("genotypes",population)
np.save("fitnesses",fitnesses)

p.disconnect()