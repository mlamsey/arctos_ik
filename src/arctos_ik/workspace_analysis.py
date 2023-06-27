import numpy as np
import csv
from utils import fibonacci_sphere, generate_rotation_matrix_from_z_axis, xyz_meshgrid, ParallelProcess
from arctos_ik import ArctosIK

def print_green(msg, end='\n'):
    print(f"\033[92m{msg}\033[0m", end=end)

def get_number_of_reachable_orientations_at_point(ik_object: ArctosIK, position, n_orientation_vectors=100, n_yaws=25, initial_configuration=np.zeros(6)):
    """
    Computes the number of reachable orientations at a given point.
    
    Reference: Zacharias, Franziska, et al. "Using a model of the reachable workspace
    to position mobile manipulators for 3-d trajectories." 2009 9th IEEE-RAS International
    Conference on Humanoid Robots. IEEE, 2009.

    Inputs: ik_object: ArctosIK object
            position: 1x3 np array XYZ coordinate (meters)
            n_orientation_vectors: int number of orientation vectors to test
            (sampled from a fibonacci sphere)
            n_yaws: int number of yaw angles to test
            initial_configuration: 1x6 np array initial guess for IK solver (radians)
    Outputs: n_valid_orientations: int number of valid orientations
    """
    
    # initialize
    n_valid_orientations = 0
    
    # sample rotations
    orientation_points = fibonacci_sphere(n_orientation_vectors)
    yaw_angles = np.linspace(-np.pi, np.pi, n_yaws)

    # iterate through all orientation vectors
    for orientation_vector in orientation_points:
        R = generate_rotation_matrix_from_z_axis(orientation_vector)
        bool_reachable = False

        # iterate through all yaw angles
        for yaw in yaw_angles:
            z_rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                            [np.sin(yaw), np.cos(yaw), 0],
                                            [0, 0, 1]])
            R_new = np.matmul(R, z_rotation_matrix)

            # solve IK
            _, (pos_error, ori_error) = ik_object.ik(target_position=position,
                                    target_orientation=R_new,
                                    initial_guess=initial_configuration,
                                    pos_error_threshold=10.,
                                    ori_error_threshold=10.)

            # pos error threshold: 1cm; ori error threshold: 0.05 rad
            if pos_error < 0.01 and ori_error < 0.05:
                bool_reachable = True
                continue
        
        if bool_reachable:
            n_valid_orientations += 1

    return n_valid_orientations

def parallel_reachability(points, n_orientation_vectors=100, n_yaws=20, n_parallel_workers=12):
    """
    Computes the reachability scores for a given set of points (uses parallel processing)
    Inputs: points: Nx3 np array XYZ coordinates (meters)
            n_orientation_vectors: int number of orientation vectors to test
            (sampled from a fibonacci sphere)
            n_yaws: int number of yaw angles to test
    Outputs: scores: list of reachability scores
    """

    # init
    parallel_process = ParallelProcess()
    ik_object = ArctosIK()

    def parallel_operation(idx, position):
        result = get_number_of_reachable_orientations_at_point(ik_object,
                                                               position,
                                                               n_orientation_vectors=n_orientation_vectors,
                                                               n_yaws=n_yaws,
                                                               initial_configuration=np.zeros(6))
        print_green(f"Point {idx}: {position} has {result} valid orientations")
        return result
    
    results = parallel_process.start(
        callback=parallel_operation,
        inputs=points,
        num_workers=n_parallel_workers
    )

    scores = [r[1] for r in results]

    return scores

def workspace_reachability(points, n_orientations=100, filename="workspace.csv"):
    """
    Computes the reachability scores for a given set of points
    Inputs: points: Nx3 np array XYZ coordinates (meters)
            n_orientations: int number of orientation vectors to test
            (sampled from a fibonacci sphere)
            filename: str filename to save data to
    Outputs: None
    """

    # init file i/o
    try:
        f = csv.writer(open(filename, "x"), delimiter=',')
        f.writerow(["x", "y", "z", "n_valid_orientations / " + str(n_orientations)])
    except Exception as e:
        print(e)
        return
    
    # compute scores and write to file
    scores = parallel_reachability(points, n_orientation_vectors=n_orientations, n_yaws=20)

    for position, score in zip(points, scores):
        f.writerow([position[0], position[1], position[2], score])

def xy_plane_workspace(z=0.3, n_orientations=100, filename="parallel_xy_plane_workspace.csv"):
    """
    Computes the reachability map for an XY plane at a given Z coordinate
    Inputs: z: float Z coordinate (meters)
            filename: str filename to save data to
    """
    
    # workspace points
    points = xyz_meshgrid(
        xlim=[0., 0.75],  # symmetry about y axis
        ylim=[-0.75, 0.75],
        zlim=[z, z],
        n_x=25,
        n_y=50,
        n_z=1
    )

    workspace_reachability(points, n_orientations=n_orientations, filename=filename)

def xz_plane_workspace(y=0., n_orientations=100, filename="parallel_xz_plane_workspace.csv"):
    """
    Computes the reachability map for an XY plane at a given Z coordinate
    Inputs: z: float Z coordinate (meters)
            filename: str filename to save data to
    """

    # init ik stuff
    points = xyz_meshgrid(
        xlim=[0, 0.75],  # symmetry about y axis
        ylim=[y, y],
        zlim=[-0.5, 1.],
        n_x=25,
        n_y=1,
        n_z=50
    )

    workspace_reachability(points, n_orientations=n_orientations, filename=filename)

def yz_plane_workspace(x=0., n_orientations=100, filename="parallel_yz_workspace.csv"):
    """
    Computes the reachability map for an XY plane at a given Z coordinate
    Inputs: z: float Z coordinate (meters)
            filename: str filename to save data to
    """

    # init ik stuff
    points = xyz_meshgrid(
        xlim=[x, x],  # symmetry about y axis
        ylim=[-0.75, 0.75],
        zlim=[-0.5, 1.],
        n_x=1,
        n_y=50,
        n_z=50
    )

    workspace_reachability(points, n_orientations=n_orientations, filename=filename)

if __name__ == '__main__':
    # xy_plane_workspace()
    # xz_plane_workspace()
    yz_plane_workspace()