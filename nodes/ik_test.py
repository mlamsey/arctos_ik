#!/usr/bin/env python

import rospy
from urdf_parser_py.urdf import URDF
import numpy as np
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from arctos_ik.arctos_ik import ArctosIK

############################################################
# Helper functions
############################################################

def make_marker(xyz, marker_size=0.05):
    """
    Create a marker message
    Inputs: xyz: 1x3 np array XYZ coordinate (meters)
            marker_size: float size of the marker (meters)
    Outputs: marker: Marker message
    """

    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = 'base_link'  # Set the frame ID
    marker.type = Marker.SPHERE
    marker.pose.position.x = xyz[0]
    marker.pose.position.y = xyz[1]
    marker.pose.position.z = xyz[2]
    marker.scale.x = marker_size
    marker.scale.y = marker_size
    marker.scale.z = marker_size
    marker.color.a = 1.0  # Fully opaque
    marker.color.r = 1.0  # Red color
    marker.color.g = 0.0
    marker.color.b = 0.0

    return marker

def get_joint_names():
    """
    Get the joint names from the current robot_description rosparam
    Outputs: joint_names: list of strings of joint names
    """

    # Get the URDF from the parameter server
    robot_urdf = rospy.get_param('robot_description')

    # Parse the URDF
    robot = URDF.from_xml_string(robot_urdf)

    # Get the joint names
    joint_names = [joint.name for joint in robot.joints]

    return joint_names

############################################################
# Main function
############################################################

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
