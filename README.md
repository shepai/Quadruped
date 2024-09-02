# Quadruped
URDF file and demo code for using a simulated quadruped robot with hobby servos and aluminium parts. 

<img width=70% src="https://github.com/shepai/Quadruped/blob/main/assets/diagramQuadruped2.png?raw=true">

## Robots
This repository contains different variations of the quadruped robot. 

To import these you will need to select the correct file

```python
path="C:/Users/.../Quadruped/Quadruped_sim/urdf/"

flags = p.URDF_USE_SELF_COLLISION
robot_id = p.loadURDF(path+"Quadruped.urdf",flags=flags)
```

If you want attached round feet then you can use:

```python
path="C:/Users/.../Quadruped/Quadruped_sim/urdf/"

flags = p.URDF_USE_SELF_COLLISION
robot_id = p.loadURDF(path+"Quadruped_prestip.urdf",flags=flags)
```

The press tip is a tactile sensor. This does not work in simulation but provides a hard outside. 