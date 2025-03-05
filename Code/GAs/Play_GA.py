if __name__=="__main__":
    import sys
    sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
from environment import *
from CPG import *
import pickle
with open('/its/home/drs25/Documents/GitHub/Quadruped/models/genotypes_8.pkl', 'rb') as f:
    population = pickle.load(f)
fitnesses=np.load("/its/home/drs25/Documents/GitHub/Quadruped/models/fitnesses_8.npy")
def fitness_(robot,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        #calculate the phase length of the hip
        """oscillations = np.diff(np.array(history['motors'])[:,3])
        zero_crossings = np.where(np.diff(np.sign(oscillations)) != 0)[0] + 1  # +1 to correct index shift
        diff=np.diff(zero_crossings)
        if type(np.diff(zero_crossings))==type([]) or type(np.diff(zero_crossings))==type(np.array([])): 
            if len(np.diff(zero_crossings))>0:diff=np.average(diff)
            else: diff=0
        fitness+=diff/10 #more phase is betters"""
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

#find best fitness
best=0
index=0
env=environment(0,0)
print("\n\n\n\n\n\n\n")
for i in range(len(fitnesses)): #
    print("------------>",i)
    env.reset()
    fit,mot=env.runTrial(population[i],50,delay=0,fitness=fitness_)
    if fit>best:
        best=fit
        index=i

env.close()
print(index)
env=environment(True,1,"/its/home/drs25/Documents/GitHub/Quadruped/assets/videos/example_8.mp4")
fit,mot=env.runTrial(population[index],100,delay=1,fitness=fitness_)
env.stop()
np.savez("/its/home/drs25/Documents/GitHub/Quadruped/Code/GAs/motors_8",mot,allow_pickle=True)