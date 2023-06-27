#!/usr/bin/env python

import rospy
import rospkg
from visualization_msgs.msg import MarkerArray
import csv
from arctos_ik.utils_ros import make_marker
from matplotlib.cm import get_cmap

class Point:
    def __init__(self, x, y, z, score=0.) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.score = score
    
    def get_point(self):
        return [self.x, self.y, self.z]
    
    def get_score(self):
        return self.score

    def get_point_score_tuple(self):
        return (self.get_point(), self.score)

class WorkspaceMarkerBroadcaster:
    def __init__(self) -> None:
        self.workspace_data = []
        self.workspace_marker_publisher = rospy.Publisher('workspace_markers', MarkerArray, queue_size=10)

    def load_workspace_markers(self, csv_file_path) -> None:
        """
        Loads the workspace markers from a csv file
        """

        try:
            # load csv
            reader = csv.reader(open(csv_file_path, newline=''), delimiter=',')
            
            # get data
            points = []
            for row in reader:
                if row[0] == "x":
                    continue

                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
                n_valid_orientations = int(row[3])
                points.append(Point(x, y, z, n_valid_orientations))
        except Exception as e:
            print(e)
            return
        
        # save data
        self.workspace_data = points

    def make_workspace_marker_array(self) -> MarkerArray:
        """
        Creates a MarkerArray message from the workspace data
        Output: MarkerArray message
        """
        
        # config
        marker_size = 0.05
        n_orientations = 100.
        
        # init
        array = MarkerArray()
        cmap = get_cmap('jet')
        i = 0
        for point in self.workspace_data:
            if point.get_score() > 0:
                marker = make_marker(point.get_point(), marker_size=marker_size)
                marker.id = i

                # color
                color = cmap(1. - (point.score / n_orientations))
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.8
                array.markers.append(marker)
                i += 1

                # add mirrored marker
                if abs(point.x) > 0.001:
                    marker = make_marker([-point.x, point.y, point.z], marker_size=marker_size)
                    marker.id = i

                    # color
                    marker.color.r = color[0]
                    marker.color.g = color[1]
                    marker.color.b = color[2]
                    marker.color.a = 0.8
                    array.markers.append(marker)
                    i += 1

        return array

    def main(self) -> None:
        rospy.init_node('workspace_markers')
        rate = rospy.Rate(10) # 10hz

        # load workspace markers
        marker_data_path = f"{rospkg.RosPack().get_path('arctos_ik')}/data/xy_plane_workspace.csv"
        self.load_workspace_markers(marker_data_path)

        while not rospy.is_shutdown():
            marker_array = self.make_workspace_marker_array()
            self.workspace_marker_publisher.publish(marker_array)
            rate.sleep()

if __name__ == '__main__':
    WorkspaceMarkerBroadcaster().main()