import numpy as np
import multiprocessing
from typing import List, Callable

############################################################
# Methods
############################################################
def fibonacci_sphere(samples=1) -> np.ndarray:
    """
    Generate points on a sphere using the Fibonacci spiral method.
    Input: samples - number of points to generate: N
    Output: points - numpy array of points: N x 3 (x, y, z)
    """

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y*y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def generate_rotation_matrix_from_z_axis(z_axis) -> np.ndarray:
    """
    Generates a rotation matrix from a given z-axis
    The rotation about the z-axis is arbitrary
    Input: z_axis - 3x1 np array
    Output: R - 3x3 np array rotation matrix
    TODO: check if right handed
    """
    # x_axis = np.array([z_axis[1], -z_axis[0], 0])

    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.random.rand(3)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.vstack((x_axis, y_axis, z_axis)).T

def xyz_meshgrid(xlim, ylim, zlim, n_x=10, n_y=10, n_z=10) -> np.ndarray:
    """
    Creates a meshgrid of points in 3D space
    Inputs: xlim - 1x2 np array x-axis limits
            ylim - 1x2 np array y-axis limits
            zlim - 1x2 np array z-axis limits
            n_x - int number of points in x-axis
            n_y - int number of points in y-axis
            n_z - int number of points in z-axis
    Output: xyz_points - nx3 np array of points
            (n = n_x * n_y * n_z)
    """

    x_points = np.linspace(xlim[0], xlim[1], n_x)
    y_points = np.linspace(ylim[0], ylim[1], n_y)
    z_points = np.linspace(zlim[0], zlim[1], n_z)
    X, Y, Z = np.meshgrid(x_points, y_points, z_points)
    xyz_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    return xyz_points

def quaternion_angular_distance(u, v) -> float:
    """
    Computes the angular distance between two quaternions
    Quaternion format: [w, x, y, z]
    Inputs: u - 1x4 np array quaternion
            v - 1x4 np array quaternion
    Output: angular_distance - float angular distance in radianss
    """

    # Ensure u and v are numpy arrays
    u = np.array(u)
    v = np.array(v)
    
    # Compute the dot product of the two quaternions
    dot_product = np.dot(u, v)
    
    # Compute the angular distance in radians
    if np.abs(dot_product) > 1:
        angular_distance = np.pi
    else:
        angular_distance = 2 * np.arccos(np.abs(dot_product))
    
    return angular_distance

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Inputs: R: 3x3 rotation matrix
    Outputs: Quaternion as [w, x, y, z]
    """
    # Ensure R is a numpy array
    R = np.array(R)
    
    # Compute the trace of the matrix
    trace = np.trace(R)
    
    # Depending on the trace, calculate the quaternion components
    if trace > 0:
        w = np.sqrt(1 + trace) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        # Find the index of the largest diagonal element
        d = np.diagonal(R)
        max_diag_index = np.argmax(d)
        
        # Compute quaternion components depending on the index
        if max_diag_index == 0:
            S = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif max_diag_index == 1:
            S = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    
    # Return quaternion
    return [w, x, y, z]

############################################################
# Objects
############################################################
class ParallelProcess:
    def __init__(self):
        self.result_map = {}

    def __worker(self, queue, callback, results):
        while True:
            item = queue.get()  # Get an item from the queue
            if item is None:
                break  # Break the loop when there are no more items
            # Perform your operation on the item
            result = callback(item[0], item[1])
            results.append((item[0], result))

    def get_result_map(self):
        return self.result_map

    def start(self,
              callback: Callable,
              inputs: List,
              num_workers: int = 8) -> List:
        """
        Start a parallel process. The callback function should take two
        arguments: the index of the input and the input itself. The callback
        then returns the result of the operation on the input.
        """
        # Create a multiprocessing queue
        queue = multiprocessing.Queue()

        # Create a shared list for storing the results
        manager = multiprocessing.Manager()
        results = manager.list()

        # Create worker processes
        processes = []
        for _ in range(num_workers):
            process = multiprocessing.Process(
                target=self.__worker,
                args=(queue, callback, results))
            process.start()
            processes.append(process)

        # Put inputs into the queue
        for i, input in enumerate(inputs):
            queue.put((i, input))

        # Add None items to indicate the end of the queue
        for _ in range(num_workers):
            queue.put(None)

        # Wait for all worker processes to finish
        for process in processes:
            process.join()

        # Sanity check to ensure that all items have been processed
        assert queue.empty()
        assert len(results) == len(inputs)
        results = sorted(results, key=lambda x: x[0])
        return list(results)
    
if __name__ == '__main__':
    # Test Z axis thing
    for _ in range(100):
        z_axis = np.random.randn(3)
        R = generate_rotation_matrix_from_z_axis(z_axis)
        d0 = np.dot(R[:, 0], R[:, 1])
        d1 = np.dot(R[:, 0], R[:, 2])
        d2 = np.dot(R[:, 1], R[:, 2])
        if d0 < 10.e9 and d1 < 10.e9 and d2 < 10.e9:
            print("Orthogonal")
        else:
            print("Not orthogonal")
            print(d0)
            print(d1)
            print(d2)
            print(R)
            print(z_axis)
            
    # R = generate_rotation_matrix_from_z_axis(np.array([1., 0., 1.]))
    # print(R)
    # # print(R[:, 0])
    # print(np.dot(R[:, 0], R[:, 1]))
    # print(np.dot(R[:, 0], R[:, 2]))
    # print(np.dot(R[:, 1], R[:, 2]))

    # # Test Fibonacci sphere
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # from mpl_toolkits.mplot3d import Axes3D
    # orientation_points = fibonacci_sphere(100)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(orientation_points[:, 0], orientation_points[:, 1], orientation_points[:, 2])
    # plt.show()

    # # Test rotation computations
    # R1 = [[0, -1, 0],
    #     [1, 0, 0],
    #     [0, 0, 1]]

    # R2 = [[1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]]

    # q1 = rotation_matrix_to_quaternion(R1)
    # q2 = rotation_matrix_to_quaternion(R2)

    # print(quaternion_angular_distance(q1, q2))
    