import rospy
from visualization_msgs.msg import Marker
from urdf_parser_py.urdf import URDF

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
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker_size
    marker.scale.y = marker_size
    marker.scale.z = marker_size
    marker.color.a = 1.0  # Fully opaque
    marker.color.r = 1.0  # Red color
    marker.color.g = 0.0
    marker.color.b = 0.0

    return marker