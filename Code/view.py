from CPG import *
import matplotlib.pyplot as plt

agent=CPG(6)
#agent=Network(6,std=1.5)
inputs=np.zeros((3,))
ar=[]
for i in range(10000):
    out=agent.get_positions(inputs)
    #out=agent.forward()
    ar.append(out)

ar=np.array(ar)
leg1=ar[:,0:3]

# Set up the figure and two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot data on the first subplot
ax1.plot(leg1[:,0],leg1[:,1])
ax1.set_title("Leg positions")
ax1.set_xlabel("knee position")
ax1.set_ylabel("hip position")
ax1.legend()

# Plot data on the second subplot
ax2.plot(ar[:,0:3])
ax2.set_title("CTRNN output")
ax2.set_xlabel("time")
ax2.set_ylabel("neuron output")
ax2.legend()

# Display the plots
plt.tight_layout()
plt.show()

