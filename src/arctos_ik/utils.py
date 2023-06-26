import numpy as np
import multiprocessing
from typing import List, Callable

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
    """

    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.array([z_axis[1], -z_axis[0], 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.vstack((x_axis, y_axis, z_axis)).T

def xyz_meshgrid(xlim, ylim, zlim, n_x=10, n_y=10, n_z=10) -> np.ndarray:
    x_points = np.linspace(xlim[0], xlim[1], n_x)
    y_points = np.linspace(ylim[0], ylim[1], n_y)
    z_points = np.linspace(zlim[0], zlim[1], n_z)
    X, Y, Z = np.meshgrid(x_points, y_points, z_points)
    xyz_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    return xyz_points

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