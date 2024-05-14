This repository creates a model predictive contouring controller to use on a f110th racestack.

<img src="https://github.com/wueestry/f110_mpcc/blob/main/docs/MPC_sim.gif" width="700" />
 (source: https://github.com/alexliniger/MPCC)

\
The main branch is using a pacejka tire model to simulate the cars behaviour, while the additional branch `simplified_tire_model` uses a simpler, path-parametric tire model

# Setup
- install ros-noetic and f110 base system
- clone repo into catkin ws
- install python requirements by running `pip install -r requirements.txt` 
- use `catkin build` to compile ros pkgs
- start mpcc node by running `roslaunch mpcc_ros mpcc_controller.launch`

# Obstacle Avoidance Demo
<img src="https://github.com/wueestry/f110_mpcc/blob/main/docs/obstacle_avoidance.gif" width="700" />
