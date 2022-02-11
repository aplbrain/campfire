""" This script, when run, will convolve each slice of a given 3D binary volume such that
the result are vectors pointing away from the walls, looking at range mem_radius
"""

import os
import numpy as np
from scipy.ndimage.filters import convolve

def load_membrane_vectors(precomp_fn):
    # If there already is a saved out precompute file, just load it!
    temp = np.load(precomp_fn)
    data = temp["data"]
    return data

def precompute_membrane_vectors(mem, memz, precomp_fn=None, mem_radius=3):
    """
    Convolve membrane with each directional sensor and then
    concatenate channels to form a lookup table
    This will be equivalent to a sensor sampling within a given radius
    and computing point-wise influence vectors
    :param mem: data matrix (edge filtered)
    :param memz: data martix, edge filtered along z axis
    :param comp_flag: Whether or not to precompute or load
    :param precomp_fn: location of saved-out precompute file
    :return:
    """

    rad = int(mem_radius / 2)
    a = np.arange(mem_radius) - rad
    a[:rad], a[-rad:] = np.flip(a[:rad]), np.flip(a[-rad:])
    kx, ky, kz = np.meshgrid(a, a, a)

    mem_x = convolve(mem, kx, mode="constant", cval=5)[..., np.newaxis]
    mem_y = convolve(mem, ky, mode="constant", cval=5)[..., np.newaxis]
    # Note memz is used here - the yz plane edge detection is important
    mem_z = convolve(memz, kz, mode="constant", cval=5)[..., np.newaxis]

    data = np.concatenate((mem_x, mem_y, mem_z), axis=-1)
    if precomp_fn:
        np.savez(precomp_fn, data=data)
    return data

# Create a queue out of any of the above sampling methods
def create_queue(data_shape, n_pts, sampling_type="lin", syns=None, isotropy=10):
    """
    Create spawn position list based on the desired sampling method

    Arguments:
        data_shape: (x,y,z) shape of the image / membrane data
        n_pts:  number of points per dimension to spawn linearly
        sampling type: "lin" or "synapse" to switch modes
        syns: for synapse spawning, the IDS to spawn at

    Returns:
        q: list of positions to spawn at
    """
    if sampling_type == "lin":
        x_coords = np.round(
            np.linspace(0.025 * data_shape[0], 0.975 * data_shape[0], n_pts)
        )
        y_coords = np.round(
            np.linspace(0.025 * data_shape[1], 0.975 * data_shape[1], n_pts)
        )
        z_coords = np.round(
            np.linspace(0.025 * data_shape[2], 0.975 * data_shape[2], n_pts // isotropy)
        )

        q = []
        for x in range(n_pts):
            for y in range(n_pts):
                for z in range(n_pts // isotropy):
                    q.append([x_coords[x], y_coords[y], z_coords[z]])

    if sampling_type == "extension":
        # TODO 
        q = spawn_from_neuron_extension(syns)
    return q

def spawn_from_neuron_extension(neurons, n_pts_per_neuron, ids=None):
    """
    Generate a set of agent spawning points given a volume of ground truth neurons
    :param neurons: Numpy array of neurons (with integer IDs)
    :param n_pts_per_neuron: Number of points to sample
    :param ids: [Optional] List of ids from which to downselect
    :return:

    TODO flesh out this old function for the extension task
    """
    if ids is not None:
        unique_neurons = ids
    else:
        unique_neurons = np.unique(neurons.flatten())
        unique_neurons = unique_neurons[unique_neurons > 0]

    x, y, z = np.meshgrid(
        np.arange(neurons.shape[0]),
        np.arange(neurons.shape[1]),
        np.arange(neurons.shape[2]),
    )

    pts = []
    for neuron_id in unique_neurons:
        xn, yn, zn = (
            x[neurons == neuron_id],
            y[neurons == neuron_id],
            z[neurons == neuron_id],
        )

        idx = np.random.choice(np.arange(xn.flatten().shape[0]), n_pts_per_neuron)
        for ind in idx:
            start_pt = [xn.flatten()[ind], yn.flatten()[ind], zn.flatten()[ind]]
            pts.append(start_pt)

    return pts