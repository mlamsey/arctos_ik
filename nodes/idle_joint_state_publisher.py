#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from arctos_ik.utils_ros import get_joint_names

def joint_state_publisher():
    """
    This test node publishes the joint states of the robot.
    The joint states are published on the topic '/joint_states'
    """


    # Initialize the node with rospy
    rospy.init_node('joint_state_publisher')

    # Create publisher
    joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)

    # Set a rate for the publisher
    rate = rospy.Rate(30)  # 30hz

    # Create an instance of sensor_msgs/JointState
    joint_state = JointState()
    joint_state.header = Header()
    joint_state.name = get_joint_names()  # matches joint names in URDF
    joint_state.position = [0.0 for _ in range(len(joint_state.name))]

    # While ROS is still running
    while not rospy.is_shutdown():
        # Update the header timestamp
        joint_state.header.stamp = rospy.Time.now()
            
        # Publish the joint state and target marker
        joint_pub.publish(joint_state)
        rate.sleep()

if __name__ == '__main__':
    try:
        joint_state_publisher()
    except rospy.ROSInterruptException:
        pass