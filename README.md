# Description

Quick and dirty ROS + IK for an ARCTOS robot arm.

https://arctosrobotics.com/

# Installation

This installation assumes that you have ROS installed on your system.

1. Install additional python dependencies

`pip install -r requirements.txt`

2. Clone into the robot's description (renamed here as `ruxonros2_description`)

`git clone https://github.com/ArctosRobotics/ROS ruxonros2_description`

3. Compile a URDF for the planner

`roscd ruxonros2_description`

`rosrun xacro xacro ruxonros2.xacro --inorder > ruxon.urdf`

4. Complete install

`cd` into your `catkin_ws` and run `catkin_make`

# Usage

`roslaunch arctos_ik ik_test.launch`

