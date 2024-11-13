import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

alpha=0.1
h=0.3
b=0.2
k=0.5
geno_1=np.random.normal(0,0.2,(4,))
geno_2=np.random.normal(0,0.2,(4,))

def get(i,alpha,h,b,k):
    y=alpha*np.sin((i-h)/b)+k
    return y

ar_1=[]
ar_2=[]
for i in range(1000):
    ar_1.append(get(i/100,*geno_1))
    ar_2.append(get(i/100,*geno_2))

plt.plot(ar_1,ar_2)
plt.ylabel("leg pos")
plt.xlabel("knee pos")
plt.show()