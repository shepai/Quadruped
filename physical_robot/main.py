from eyes import Eyes
from Tactile_CP import I2C_Tactile
from legs import Body
import busio
import board

i2c=busio.I2C(board.SCL,board.SDA)

#run everything on same channel
eyes=Eyes(i2c)
foot=I2C_Tactile(i2c)
body=Body(i2c)

for i in range(100):
    vals=foot.read()
    print(vals)
