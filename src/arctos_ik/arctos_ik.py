import numpy as np
import ikpy.urdf.utils
import ikpy.chain
import rospkg
from typing import Tuple
from utils import quaternion_angular_distance, rotation_matrix_to_quaternion

class ArctosIK:
    def __init__(self) -> None:
        """
        Initializes IK engine
        Loads URDF from the ruxonros2_description ROS package
        See https://github.com/ArctosRobotics/ROS
        """

        self.urdf_path = f"{rospkg.RosPack().get_path('ruxonros2_description')}/urdf/ruxon.urdf"
        active_link_mask = [False] + [True] * 6
        self.ik_chain = ikpy.chain.Chain.from_urdf_file(self.urdf_path, active_links_mask=active_link_mask)

    def view_tree(self) -> None:
        """
        View the URDF joint tree using the ikpy library
        """

        tree, _ = ikpy.urdf.utils.get_urdf_tree(self.urdf_path, "base_link")
        tree.view()

    def random_pose(self) -> np.ndarray:
        """
        Generates a random pose within the joint limits
        """

        real_bounds = [link.bounds for link in self.ik_chain.links][1:]  # exculde base limits (-inf, inf)
        pose = []
        for bounds in real_bounds:
            pose.append(np.random.uniform(low=bounds[0], high=bounds[1]))
        
        return np.array(pose)
    
    def center_pose(self) -> np.ndarray:
        """
        Generates a pose at the center of the joint limits
        """

        real_bounds = [link.bounds for link in self.ik_chain.links][1:]  # exculde base limits (-inf, inf)
        pose = []
        for bounds in real_bounds:
            pose.append((bounds[0] + bounds[1]) / 2)
        
        return np.array(pose)

    def ik(self, target_position: np.ndarray, target_orientation: np.ndarray=np.eye(3), initial_guess=None, pos_error_threshold=0.05, ori_error_threshold=0.05) -> Tuple:
        """
        Computes the IK solution for a given target configuration
        Inputs: target_position: 1x3 np array XYZ coordinate (meters)
                target_orientation: 3x3 np array rotation matrix (unitless)
                initial_guess: 1x6 np array initial guess for IK solver (radians)
                pos_error_threshold: float maximum error allowed for IK solution (meters)
                ori_error_threshold: float maximum error allowed for IK solution (radians)
        Outputs: q_soln: 1x6 np array joint angles (radians)
                 errors: tuple of position and orientation errors

        NOTE: you can change the guessing method by changing the initial_guess behavior below
        """

        # get initial guess
        if initial_guess is None:
            # initial_guess = self.random_pose()
            initial_guess = self.center_pose()

        # add dummy joint to initial guess (base_link is fixed)
        initial_guess = np.concatenate(([0.], initial_guess))

        # solve IK
        q_soln = self.ik_chain.inverse_kinematics(target_position=target_position,
                                                  target_orientation=target_orientation,
                                                  orientation_mode="all",
                                                  initial_position=initial_guess,
                                                  max_iter=100,)

        # calculate errors
        solution_fk = self.ik_chain.forward_kinematics(q_soln)

        solution_fk_position = solution_fk[:3, 3]
        position_error = np.linalg.norm(solution_fk_position - target_position)

        solution_fk_orientation = solution_fk[:3, :3]
        q1 = rotation_matrix_to_quaternion(solution_fk_orientation)
        q2 = rotation_matrix_to_quaternion(target_orientation)
        orientation_error = quaternion_angular_distance(q1, q2)

        # if error is too large, return zeros
        if position_error > pos_error_threshold:
            print(f"ArctosIK::ik: Position error {position_error} too large, returning zeros")
            q_soln = [0.] * 7
        
        if orientation_error > ori_error_threshold:
            print(f"ArctosIK::ik: Orientation error {orientation_error} too large, returning zeros")
            q_soln = [0.] * 7

        return q_soln[1:], (position_error, orientation_error)
