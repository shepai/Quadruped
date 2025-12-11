from scipy.signal import find_peaks
import numpy as np 
from copy import deepcopy
def breakdown(sample): #break down the wave characteristics 
    a=np.min(sample)
    b=np.max(sample)
    med=np.median(sample)
    di = 1 if abs(abs(med)-abs(a)) < abs(abs(med)-abs(b)) else -1 #if the median is closer to one it is the base
    h = abs(abs(a)-abs(b))
    if di==1: #gather where there are peaks
        base=np.where(sample>a)[0] #gather all values that are in the peak
        wid=np.where(sample<=a)[0] #gather all values that are in the base
    else: #if the wave goes down and not up
        base=np.where(sample<b)[0] #gather all values that are in the peak
        wid=np.where(sample>=b)[0] #gather all values that are in the base
    level=sample[wid[0]]
    peaks, props = find_peaks(np.abs(sample), width=True, rel_height=1.0) #it is important to 
    widths = props["widths"]     # width (right-left)
    gap=np.average(widths)
    dist=np.average(np.diff(peaks))
    wid=dist-gap
    return h*di,gap,wid,level,peaks[0]
def reform(height, gap, width, level, start, repeat=1):
    #rising/peak part sampled at integer indices
    x1 = np.arange(start,start+gap)
    y1 = height * np.sin((np.pi / gap) * (x1 - start))
    x2 = np.arange(start+gap, start+gap+width)#flat base part
    y2 = np.ones_like(x2) * level
    x = np.concatenate((x1, x2))#one full cycle
    y = np.concatenate((y1, y2))
    cycles = []#repeat cycles exactly
    ys = []
    for i in range(repeat):
        offset = i * (gap + width)
        cycles.append(x + offset)
        ys.append(y)
    x = np.concatenate(cycles)
    y = np.concatenate(ys)
    # shift so first peak is at index "start"
    peaks, _ = find_peaks(np.abs(y))
    shift = peaks[0] - start #find difference between our generated signal
    x-=start
    y = np.roll(y, -shift) #roll it round so it alignes with the start position
    return x, y

if __name__=="__main__":
    import matplotlib.pyplot as plt 
    import matplotlib 
    matplotlib.use('TkAgg')
    #load in data 
    X=np.load("/its/home/drs25/Quadruped/Code/UBERMODEL/X_DATA_o.npy")
    #motors_current,vector,feet,friction_value,orientation
    idx = np.r_[12:19, 20:X.shape[2]]
    continous=X[:,:,idx]
    idx = np.r_[0:12, 19]
    steady=X[:,:,idx]
    #loop through data 
    new_x=[]
    new_y=[]
    for i in range(len(X)):
        print("Current length:",len(new_x))
        motors=[]
        friction=steady[i,0,-1]
        trial=steady[i,:,:-1]
        error=False
        for signal in trial.T:
            try:
                h,gap,wid,level,start=breakdown(signal)#apply signal decontruction
                motors.append([h,gap,wid,level,start,friction])
            except IndexError:
                error=True
        if not error:
            new_x.append(deepcopy(motors))
            new_y.append(continous[i])

    #save signals
    print("Signals converted",len(new_x))
    np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_X",np.array(new_x))
    np.save("/its/home/drs25/Quadruped/Code/UBERMODEL/data/steady_y",np.array(new_y))