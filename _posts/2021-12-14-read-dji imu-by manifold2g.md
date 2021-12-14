---
tags: DJI
---
# read imu data by dji manifold2G
## Download related software 
- DJI assiatent2
- Onbordsdk (recommend version **3.7**)
- Onbordskk-ROS (**optional**)
## Envoriment
- ubuntu 16.04LTS
- ROS -kinetic
## hardware
- Manifold2G
- DJI A3
## process
>regest your DJI developer acount 

>create you own app

>using remdme file

>run 

```ros
cd catkin_ws
source ./devel/setup.bash
roslaunch dji_sdk sdk.launch
```
open other terminal
```ros
cd catkin_ws
source ./devel/setup.bash
rostopic echo /dji_sdk/imu
```