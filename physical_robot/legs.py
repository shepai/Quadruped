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
        start=[130.0, 140.0, 50.0, 180.0, 180.0, 65.0,110.0, 40.0, 120.0, 30,100,80]

        for i in range(self.start,num):
            self.kit.servo[i].angle=start[i]
    def reset(self):
        reset_array=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in range(12):
            self.kit.servo[i].angle=reset_array[i]
    def schedule_move(self, target_angles, step_size=1, delay=0.01):
        # Flag to track if any servo still needs to move
        moving = True

        while moving:
            moving = False  # Assume all servos have reached their target for this iteration

            for i, target_angle in enumerate(target_angles):
                #print(i)
                current_angle = self.kit.servo[i].angle or 0 # Handle case where angle is None

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
                    self.kit.servo[i].angle = new_angle

            time.sleep(delay)  # Add a delay between steps
        
            
    def move(self,motors):
        for i in range(self.start,self.num):
            self.kit.servo[i].angle=motors[i-self.start]
        #print(time.time()-t,"seconds")
    
"""        
b=body()
b.reset()
b.schedule_move([10,40,10,30,5,10,10,0,0,0,0,0,0,0,0,0])
"""
        