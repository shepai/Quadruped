import board
import busio
import adafruit_ht16k33.matrix

class Eyes:
    def __init__(self,i2c=None):
        if type(i2c)==type(None):
            self.i2c=busio.I2C(board.SCL,board.SDA)
        else:
            self.i2c=i2c
        self.matrix = adafruit_ht16k33.matrix.Matrix16x8(self.i2c)
        self.matrix.brightness=0.5
    def eyes(self):
        ar=[[0,0,1,1,1,1,0,0],
            [0,1,0,0,0,0,1,0],
            [1,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,1],
            [1,0,0,0,1,1,0,1],
            [1,0,0,0,1,1,0,1],
            [0,1,0,0,0,0,1,0],
            [0,0,1,1,1,1,0,0],
            [0,0,1,1,1,1,0,0],
            [0,1,0,0,0,0,1,0],
            [1,0,0,0,1,1,0,1],
            [1,0,0,0,1,1,0,1],
            [1,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,1],
            [0,1,0,0,0,0,1,0],
            [0,0,1,1,1,1,0,0]]
        
        return ar
    def show(self):
        face=self.eyes()
        for i in range(16):
            for j in range(8):
                self.matrix[i,j]=face[i][j]
        self.matrix.show()
        
e=Eyes()
e.show()