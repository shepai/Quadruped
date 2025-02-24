if __name__=="__main__":
    import sys
    sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
from environment import *
from CPG import *
import pickle
with open('genotypes.pkl', 'rb') as f:
    population = pickle.load(f)
fitnesses=np.load("fitnesses.npy")

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
def fitness(robot):
    distance = (euclidean_distance(robot.getPos()[0:2],robot.start[0:2])*0.8 - euclidean_distance(robot.start_orientation,robot.getOrientation())*0.1)-robot.get_self_collision_count()*0.01
    if distance<0: distance=0
    if robot.hasFallen(): return 0
    return distance

env=environment(True,True,"/its/home/drs25/Documents/GitHub/Quadruped/assets/videos/example.mp4")
env.runTrial(population[np.where(fitnesses==np.max(fitnesses))[0][0]],500,delay=1)
env.stop()