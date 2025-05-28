import numpy as np
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2,axis=1))
def F1(robot,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        distances=np.array(history['positions'])-np.array([robot.start])#euclidean_distance(np.array(history['positions']),np.array([robot.start]))
        distancesX=distances[-1][0]
        distancesY=distances[-1][1]
        distancesZ=distances[-1][2]
        fitness+=distancesX - (distancesY+distancesZ)/10 #np.sum(distances)
    if robot.hasFallen(): fitness=-1000
    if type(fitness)!=type(0): 
        try:
            if type(fitness)==type([]): fitness=float(fitness[0])
            else:fitness=float(fitness)
        except:
            print("shit",fitness,np.array(history['motors']).shape,np.array(history['positions']).shape,np.array(history['orientations']).shape)
            fitness=-1000
    return fitness

def F2(robot,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        alpha=10
        beta=0.05
        gamma=0.5
        positions = np.array(history['positions'])  # Shape (T, 3)
        X, Y, Z = positions[:, 0], positions[:, 1], positions[:, 2]
        forward_movement = np.abs(X[-1] - X[0])
        deviation_penalty = np.sum(np.sqrt(Y**2 + Z**2))
        velocities = np.diff(positions, axis=0)
        smoothness_penalty = np.sum(np.linalg.norm(np.diff(velocities, axis=0), axis=1))
        #print(forward_movement, deviation_penalty, smoothness_penalty)
        fitness = (alpha * forward_movement) - (beta * deviation_penalty) - (gamma * smoothness_penalty)
    if robot.hasFallen(): fitness=-1000
    if type(fitness)!=type(0): 
        try:
            if type(fitness)==type([]): fitness=float(fitness[0])
            else:fitness=float(fitness)
        except:
            print("shit",fitness,np.array(history['motors']).shape,np.array(history['positions']).shape,np.array(history['orientations']).shape)
            fitness=-1000
    return fitness

def F3(robot,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        positions = np.array(history['positions'])
        Y, X, Z = positions[:, 0], positions[:, 1], positions[:, 2]
        orientations=history['orientations']
        magnitude=np.sqrt(np.sum(np.square(orientations),axis=1))
        forward_movement = np.abs(X[-1] - X[0])
        fitness=np.abs(np.sum(np.diff(Y))) - np.sum(np.abs(magnitude))/10
    if robot.hasFallen(): fitness=-1000
    if type(fitness)!=type(0): 
        try:
            if type(fitness)==type([]): fitness=float(fitness[0])
            else:fitness=float(fitness)
        except:
            print("shit",fitness,np.array(history['motors']).shape,np.array(history['positions']).shape,np.array(history['orientations']).shape)
            fitness=-1000
    return fitness

def fitness_(robot,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        alpha=100
        beta=0.05
        gamma=0.5
        #F1
        """distances=np.array(history['positions'])-np.array([robot.start])#euclidean_distance(np.array(history['positions']),np.array([robot.start]))
        distancesX=distances[-1][0]
        distancesY=distances[-1][1]
        distancesZ=distances[-1][2]
        fitness+=distancesX - (distancesY+distancesZ)/10 #np.sum(distances)"""
        #F2
        """positions = np.array(history['positions'])  # Shape (T, 3)
        X, Y, Z = positions[:, 0], positions[:, 1], positions[:, 2]
        forward_movement = np.abs(X[-1] - X[0])
        deviation_penalty = np.sum(np.sqrt(Y**2 + Z**2))
        velocities = np.diff(positions, axis=0)
        smoothness_penalty = np.sum(np.linalg.norm(np.diff(velocities, axis=0), axis=1))
        #print(forward_movement, deviation_penalty, smoothness_penalty)
        fitness = (alpha * forward_movement) - (beta * deviation_penalty) - (gamma * smoothness_penalty)"""
        #F3
        positions = np.array(history['positions'])
        Y, X, Z = positions[:, 0], positions[:, 1], positions[:, 2]
        orientations=history['orientations']
        magnitude=np.sqrt(np.sum(np.square(orientations),axis=1))
        forward_movement = np.abs(X[-1] - X[0])
        fitness=np.abs(np.sum(np.diff(Y))) - np.sum(np.abs(magnitude))/10
    if robot.hasFallen(): fitness=-1000
    if type(fitness)!=type(0): 
        try:
            if type(fitness)==type([]): fitness=float(fitness[0])
            else:fitness=float(fitness)
        except:
            print("shit",fitness,np.array(history['motors']).shape,np.array(history['positions']).shape,np.array(history['orientations']).shape)
            fitness=-1000
    #if fitness<0: fitness=0
    return fitness
