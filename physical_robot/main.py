from eyes import Eyes
from Tactile_CP import I2C_Tactile
from legs import Body
import busio
import board
import numpy as np
import time
i2c=busio.I2C(board.SCL,board.SDA)

#run everything on same channel
eyes=Eyes(i2c)
eyes.show()
foot=I2C_Tactile(i2c)
start=0
num=12
body=Body(i2c,num,start)
start=[130.0, 80.0, 50.0, 70.0, 180.0, 65.0,100.0, 70.0, 120.0, 30,100,80]
start=np.array(start)
#body.schedule_move(start)
body.move(start)
time.sleep(1)
start+=30
start[start>180]=180
body.schedule_move(start,4)

time.sleep(1)
start-=60
start[start<0]=0
body.schedule_move(start,4)
print(start.tolist())

print("TESTING SERVOS")

"""for i in range(12):
    print("Motor",i)
    for j in range(40):
        body.kit.servo[i].angle=40+j
        time.sleep(0.1)"""
#1 3 6 7 9 10
        
#time.sleep(4)
#start[0]=90
#body.schedule_move(start)
"""for i in range(100):
    vals=foot.read()
    print(vals)"""

#
#
#

#180
#
#

#
#
#

#
#
#