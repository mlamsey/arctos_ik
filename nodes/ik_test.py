#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from arctos_ik.arctos_ik import ArctosIK
from arctos_ik.utils_ros import make_marker, get_joint_names

def joint_state_publisher():
    """
    This test node publishes the joint states of the robot
    and a target marker for visualization in RViz.

    The target position is a sinusoidal function of time.

    The IK solution is computed using the ArctosIK class.

    The joint states are published on the topic '/joint_states'
    
    The target marker is published on the topic '/target'
    """


    # Initialize the node with rospy
    rospy.init_node('joint_state_publisher')

    # Create a publisher. This will publish messages of type JointState
    # on the topic '/joint_states'
    joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)

    # Create a target marker publisher
    target_pub = rospy.Publisher('target', Marker, queue_size=10)

    # Set a rate for the publisher
    rate = rospy.Rate(30)  # 30hz

    # Create an instance of sensor_msgs/JointState
    joint_state = JointState()

    # Initialize the Header
    joint_state.header = Header()

    # Joint names. They should match the joint names in your URDF
    joint_state.name = get_joint_names()

    # Publish initial joint state
    joint_state.header.stamp = rospy.Time.now()
    joint_state.position = [0.0 for _ in range(len(joint_state.name))]
    joint_pub.publish(joint_state)

    # IK solver
    ik_solver = ArctosIK()

    # While ROS is still running
    while not rospy.is_shutdown():
        # Update the header timestamp
        joint_state.header.stamp = rospy.Time.now()

        # create a target position
        x = -0.1 * np.sin(rospy.get_time())
        y = 0.1 * np.sin(rospy.get_time()) - 0.4
        z = 0.15 * np.cos(1.5 * rospy.get_time()) + 0.3
        
        # Compute the IK solution
        q, error = ik_solver.ik(target_position=[x, y, z])

        # update the joint state
        for i in range(len(joint_state.name)):
            joint_state.position[i] = q[i]
            
        # Publish the joint state and target marker
        joint_pub.publish(joint_state)
        target_pub.publish(make_marker([x, y, z]))

        # Sleep for a bit to maintain the desired rate
        rate.sleep()

if __name__ == '__main__':
    try:
        joint_state_publisher()
    except rospy.ROSInterruptException:
        pass
