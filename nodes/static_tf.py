#!/usr/bin/env python
import rospy
import numpy as np
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped

def create_transform_message(x, y, z, roll, pitch, yaw, parent_frame_id, child_frame_id):
    # Create a TransformStamped message
    transform = TransformStamped()

    # Fill up the header
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = parent_frame_id
    transform.child_frame_id = child_frame_id

    # Set up the translation
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.translation.z = z

    # Set up the rotation
    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    transform.transform.rotation.x = quat[0]
    transform.transform.rotation.y = quat[1]
    transform.transform.rotation.z = quat[2]
    transform.transform.rotation.w = quat[3]

    return transform

def static_tf_broadcaster():
    # Initialize the node
    rospy.init_node('static_tf_broadcaster')

    # Create tf2 broadcasters
    base_broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Loop
    rate = rospy.Rate(10.)
    while not rospy.is_shutdown():
        transform = create_transform_message(0., 0.5, 0.4, 0, 0, -np.pi/2., "base_link", "Rev28")
        base_broadcaster.sendTransform(transform)        
        rate.sleep()

if __name__ == '__main__':
    try:
        static_tf_broadcaster()
    except rospy.ROSInterruptException:
        pass
