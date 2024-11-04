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
        start=[130.0, 80.0, 50.0, 70.0, 180.0, 65.0,100.0, 70.0, 120.0, 30,100,80]

        for i in range(self.start,num):
            self.safe_set(i,start[i])
    def reset(self):
        reset_array=[130.0, 80.0, 50.0, 70.0, 180.0, 65.0,100.0, 70.0, 120.0, 30,100,80]
        for i in range(self.start,self.num):
            self.kit.servo[i].angle=reset_array[i]
    def safe_set(self,index,angle):
        if index==1: index=12
        retries=3
        for _ in range(retries):
            try:
                self.kit.servo[index].angle=angle
                return
            except OSError as e:
                print("OSERROR")
                time.sleep(0.1)
        print("failed to set servo angle")
    def safe_get(self,index):
        if index==1: index=12
        retries=3
        for _ in range(retries):
            try:
                angle=self.kit.servo[index].angle
                return angle
            except OSError as e:
                print("OSERROR")
                time.sleep(0.1)
        print("failed to set servo angle") 
    def schedule_move(self, target_angles, step_size=1, delay=0.01):
        # Flag to track if any servo still needs to move
        moving = True

        while moving:
            moving = False  # Assume all servos have reached their target for this iteration

            for i, target_angle in enumerate(target_angles):
                #print(i)
                current_angle = self.safe_get(i) or 0 # Handle case where angle is None

                # Calculate step direction based on difference
                if not(current_angle>target_angle-1 and current_angle<target_angle+1): #within bounds
                    if current_angle < target_angle:
                        new_angle = min(current_angle + step_size, target_angle)
                        moving = True
                    elif current_angle > target_angle:
                        new_angle = max(current_angle - step_size, target_angle)
                        moving = True
                    else:
                        new_angle = current_angle  # Servo is already at target

                    # Set the new angle
                    self.safe_set(i,new_angle)

            time.sleep(delay)  # Add a delay between steps
    def crawl_step(self):
        leg_seq=[[30,-30,0],[0,30,0]] #,[-30,0,0]
        bod_seq=[0,2,1,3]
        
        for j in range(len(bod_seq)):
            leg=bod_seq[j]
            for i in range(len(leg_seq)):
                if leg==0 or leg==2:
                    leg_seq[i][0]*=-1
                    leg_seq[i][1]*=-1
                motors=np.array([0 for k in range(3*leg)]+leg_seq[i]+[0 for k in range(12-len([0 for p in range(3*leg)]+leg_seq[i]))])
                seq=motors+np.array([self.safe_get(k) for k in range(self.start,self.num)])
                seq[seq<0]=0
                seq[seq>180]=180
                self.schedule_move(seq,5)
                
        self.reset()
    def move(self,motors):
        for i in range(self.start,self.num):
            self.kit.servo[i].angle=motors[i-self.start]
        #print(time.time()-t,"seconds")
    
"""        
b=body()
b.reset()
b.schedule_move([10,40,10,30,5,10,10,0,0,0,0,0,0,0,0,0])
"""
        