import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#import simulator 
from environment import *
from CPG import *
import numpy as np
import time

#create dataset writer class
dataset=np.zeros((2000,1000,12+3)) #12 motors and 3 tilt


#set up env
env=environment(0)
env.stop()

for i in range(2000):
    env.reset() #reset trial
    ar=np.random.random(((len(env.quad.motors))))*10
    env.quad.setPositions(ar)
    for j in range(1000):
        #launch trial
        print("Trial {} iteration {}".format(i, j))
        motors=env.quad.motors
        new_motors=np.array(motors) +np.random.normal(0,10,(len(motors))) #move via noise
        env.quad.setPositions(new_motors)
        #random moves
        #save moves and what happened pip install
        env.step()
        tilt=env.quad.getPos()
        dataset[i][j]=np.concatenate([new_motors,tilt])
        os.system('clear')
    np.save("/its/home/drs25/Quadruped/Code/data_collect_proj/dataset_jumbled",dataset)
