path="C:/Users/dexte/Documents/GitHub/Quadruped/Quadruped_sim/urdf/"
path="/its/home/drs25/Documents/GitHub/Quadruped/Quadruped_sim/PressTip/urdf/"
import pybullet as p
import pybullet_data
import time
import math as maths
import Quadruped
import numpy as np
import os
import random
from copy import deepcopy
# Initialize the PyBullet physics engine
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

# Define constants for the number of legs and motors
NUM_LEGS = 4
NUM_MOTORS_PER_LEG = 3  # hip, knee, ankle
NUM_PARAMS_PER_MOTOR = 4  # Amplitude, Frequency, Phase, Baseline

# Total parameters per leg
PARAMS_PER_LEG = NUM_MOTORS_PER_LEG * NUM_PARAMS_PER_MOTOR
GENOTYPE_SIZE = NUM_LEGS * PARAMS_PER_LEG + (NUM_LEGS - 1)  # 48 for motors, 3 for phase offsets

class Genotype:
    def __init__(self, params=None):
        # Initialize a random genotype if none is provided
        if params is None:
            self.params = np.random.uniform(-1, 1, GENOTYPE_SIZE)
        else:
            self.params = params

    def __getitem__(self, index):
        return self.params[index]

    def __setitem__(self, index, value):
        self.params[index] = value

    def mutate(self, mutation_rate=0.1):
        """ Mutate the genotype by tweaking random parameters """
        for i in range(len(self.params)):
            if random.random() < mutation_rate:
                self.params[i] += np.random.uniform(-0.1, 0.1)  # small random change

    def crossover(self, other):
        """ Crossover with another genotype """
        crossover_point = random.randint(0, GENOTYPE_SIZE - 1)
        child_params = np.concatenate((self.params[:crossover_point], other.params[crossover_point:]))
        return Genotype(child_params)

    def evaluate_fitness(self, environment):
        """
        Fitness function to evaluate the genotype.
        - environment: the simulation environment (e.g., PyBullet) for testing walking performance.
        """
        # Convert the genotype into CPG motor positions over time
        time_steps = np.linspace(0, 10, 100)  # Simulate 10 seconds in 100 steps
        fitness = 0.0

        for t in time_steps:
            # Get motor positions based on sine wave (CPG) for each leg
            motor_positions = self.get_motor_positions(t)
            # Step the simulation in the environment with these motor positions
            fitness += environment.step(motor_positions)  # Assume environment has a step function

        return fitness

    def get_motor_positions(self, t):
        """
        Generate motor positions using the CPG at time t.
        - t: current time step.
        Returns a list of motor positions for all 12 motors (3 per leg, 4 legs).
        """
        motor_positions = []
        for leg in range(NUM_LEGS):
            for motor in range(NUM_MOTORS_PER_LEG):
                amplitude = self.params[leg * PARAMS_PER_LEG + motor * 4 + 0]
                frequency = self.params[leg * PARAMS_PER_LEG + motor * 4 + 1]
                phase_offset = self.params[leg * PARAMS_PER_LEG + motor * 4 + 2]
                baseline_offset = self.params[leg * PARAMS_PER_LEG + motor * 4 + 3]

                # Calculate the motor position as a sine wave
                motor_position = amplitude * np.sin(frequency * t + phase_offset) + baseline_offset
                motor_positions.append(motor_position)

        # Add phase offsets between legs for coordination (3 offsets for 3 legs)
        phase_offsets = self.params[-3:]
        motor_positions[3:] = [pos + phase_offsets[i % 3] for i, pos in enumerate(motor_positions[3:])]

        return motor_positions

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def fitness(robot):
    distance = (euclidean_distance(robot.getPos()[0:2],robot.start[0:2])*0.8 - euclidean_distance(robot.start_orientation,robot.getOrientation())*0.1)-robot.get_self_collision_count()*0.01
    if distance<0: distance=0
    if robot.hasFallen(): return 0
    return distance

def runTrial(genotpye,generations,time_=100,delay=True):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF('plane.urdf')
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    initial_position = [0, 0, 0.3]  # x=1, y=2, z=0.5
    initial_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation (Euler angles to quaternion)
    flags = p.URDF_USE_SELF_COLLISION
    robot_id = p.loadURDF(path+"Quadruped_prestip.urdf", initial_position, initial_orientation,flags=flags)
    for i in range(100):
        p.stepSimulation()
        p.setTimeStep(1./24.)
    
    quad=Quadruped.Quadruped(p,robot_id,plane_id)
    quad.neutral=[-10,0,30,0,0,0,0,0,0,0,0,0]
    quad.reset()
    #GA
    TIME=0
    dt=0.01
    for i in range(generations):
        for j in range(time_):
            positions=genotpye.get_motor_positions(j)
            quad.setPositions(np.degrees(positions)//2)
            for k in range(10):
                p.stepSimulation()
                if delay: time.sleep(1./240.)
                else: p.setTimeStep(1./240.)
                basePos, baseOrn = p.getBasePositionAndOrientation(robot_id) # Get model position
                p.resetDebugVisualizerCamera( cameraDistance=0.3, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos) # fix camera onto model
                if quad.hasFallen():
                    break
        if quad.hasFallen():
            break
        TIME+=dt
    f=fitness(quad)+TIME
    del quad
    p.removeBody(robot_id)
    return f

#initial
population_size=50
population=[Genotype() for _ in range(population_size)]#np.random.choice([50, 20, 0,0,0,0,-20],(150,15,12)) #12 motors, 15 steps
fitnesses=np.zeros((population_size,))
generations=1000

#get fitnesses
for i in range(len(fitnesses)):
    fitnesses[i]=runTrial(population[i],100,delay=0)
    print(i,"/",len(fitnesses), fitnesses[i])

for gen in range(generations):
    clear = lambda: os.system('clear')
    print("Generation ",gen+1,"Best fitness",np.max(fitnesses))
    ind1=np.random.randint(0,len(fitnesses)-1)
    ind2=np.random.randint(0,len(fitnesses)-1)
    geno=population[ind1]
    f=runTrial(geno,100,delay=False) #run trial
    if f>fitnesses[ind2]: #selection
        mutated=deepcopy(geno)
        mutated.mutate()
        fitnesses[ind2]=runTrial(mutated,100,delay=False)
        population[ind2]=mutated
    else:
        mutated=deepcopy(geno)
        mutated.mutate()
        fitnesses[ind1]=runTrial(mutated,100,delay=False)
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
runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],150)
p.disconnect()