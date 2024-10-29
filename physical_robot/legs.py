import time
from adafruit_servokit import ServoKit
import numpy as np

class Body:
    def __init__(self,i2c=None,num=16,start=0):
        self.num=num
        self.start=start
        if type(i2c)==type(None):
            self.kit=ServoKit(channels=16)
        else:
            self.i2c=i2c
            self.kit=ServoKit(channels=16,i2c=self.i2c)
        start=[130.0, 10.0, 10.0, 180.0, 180.0, 65.0,110.0, 40.0, 120.0, 0,0,80]

        for i in range(self.start,num):
            self.kit.servo[i].angle=start[i]
    def reset(self):
        reset_array=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(12):
            self.kit.servo[i].angle=reset_array[i]
    def schedule_move(self,moves,step=5):
        moves=np.array(moves).astype(np.int16)
        moves[moves>180]=180
        moves[moves<0]=0
        current=np.array([int(self.kit.servo[i].angle) for i in range(self.start,self.num)]).astype(np.int16)
        diff=moves-current
        print(moves,"\n",current,"\n",diff)
        t=time.time()
        c=0
        diff[diff<2]=0
        if int(np.mean(diff))==1:
            step=1
        while np.sum(np.abs(diff))!=0:
            for i in range(self.start,self.num):
                if self.kit.servo[i].angle+(-step if diff[i-self.start]<0 else step)<=180 and self.kit.servo[i].angle+(-step if diff[i-self.start]<0 else step)>=0:
                    self.kit.servo[i].angle+= -step if diff[i-self.start]<0 else step
                    if diff[i-self.start]!=0:
                        diff[i-self.start] = diff[i-self.start]-step if diff[i-self.start]>0 else diff[i-self.start]+step
                else:
                    diff[i-self.start]=0
            c+=1
            if c>15 and step!=1: #prevent never finding 0
                step-=1
                c=0
            
    def move(self,motors):
        for i in range(self.start,self.num):
            self.kit.servo[i].angle=motors[i-self.start]
        #print(time.time()-t,"seconds")
    
"""        
b=body()
b.reset()
b.schedule_move([10,40,10,30,5,10,10,0,0,0,0,0,0,0,0,0])
"""
        