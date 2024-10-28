import time
from adafruit_servokit import ServoKit
import numpy as np

class body:
    def __init__(self,i2c=None):
        if type(i2c)==type(None):
            self.kit=ServoKit(channels=16)
        else:
            self.i2c=i2c
            self.kit=ServoKit(channels=16,i2c=self.i2c)
        
    def reset(self):
        reset_array=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(16):
            self.kit.servo[i].angle=reset_array[i]
    def schedule_move(self,moves,step=5):
        moves=np.array(moves)
        current=np.array([int(np.degrees(self.kit.servo[i].angle)) for i in range(16)])
        diff=moves-current
        t=time.time()
        c=0
        while np.sum(diff)!=0:
            for i in range(16):
                self.kit.servo[i].angle+= -step if diff[i]<0 else step
                if diff[i]!=0:
                    diff[i] = diff[i]-step if diff[i]>0 else diff[i]+step
            c+=1
            if c>15 and step!=1: #prevent never finding 0
                step-=1
                c=0
        #print(time.time()-t,"seconds")
    
        
b=body()
b.reset()
b.schedule_move([10,40,10,30,5,10,10,0,0,0,0,0,0,0,0,0])
        