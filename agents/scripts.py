import numpy as np
from scipy.ndimage.filters import convolve
import pandas as pd
from caveclient import CAVEclient

def load_membrane_vectors(precomp_fn):
    # If there already is a saved out precompute file, just load it!
    temp = np.load(precomp_fn)
    data = temp["data"]
    return data
def ver():
    print("121")

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
def create_queue(data_shape, n_pts, sampling_type="lin",
    isotropy=10, segmentation=np.zeros(1), root_id=-1):
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
        q = []

        ids, centers, sizes = get_contacts(segmentation, root_id)
        rid_size = sizes[root_id]
        client = CAVEclient('minnie65_phase3_v1')
        for i in ids:
            if get_soma(i, client) > 0:
                print("FILTERED", i)
                continue
            n_gen = int((sizes[i] / rid_size) * n_pts)
            centers_list = centers[i]
            for j in range(len(centers_list)):
                for _ in range(n_gen):
                    q.append([*centers_list[j]])
    print("Returning Queue")
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

class Intersection():
    # Code based on https://towardsdatascience.com/find-the-intersection-of-two-sets-of-coordinates-and-sort-by-colors-using-python-oop-7785f47a93b3
    def __init__ (self, sets, root_ids):
        self.sets = sets
        self.root_ids = root_ids
        self.id_list = []

    def concArrays(self):
            set_list = self.sets[0]
            self.id_list = [self.root_ids[0]]*len(self.sets[0])
            
            for i in range(1,len(self.sets)):
                set_list = set_list + self.sets[i]
                self.id_list = self.id_list + [self.root_ids[i]]*len(self.sets[i])
                
            self.sets = set_list
            return self.sets
    
    def sortList(self):


        if type(self.sets[0]) == tuple:
            sets_unpack = [(*coord, id_num) for coord, id_num in zip(self.sets,self.id_list)]
        else: # intended for np.ndarray
            sets_unpack = [(*coord.astype(int), id_num) for coord, id_num in zip(self.sets,self.id_list)]
            

        sets_unpack = sorted(sets_unpack, key=lambda x: x[2])

        sets_unpack = sorted(sets_unpack, key=lambda x: x[1])
        
        sets_unpack = sorted(sets_unpack, key=lambda x: x[0])

        self.sets = sets_unpack           

    def clash(self):
#         return [(self.sets[i][-1], self.sets[i+1][-1], *self.sets[i][:-1]) for i in range(0, len(self.sets)-1) if np.array_equal(self.sets[i][:-1], self.sets[i+1][:-1])]
        return [[self.sets[i][-1], self.sets[i+1][-1], *self.sets[i][:-1]] for i in range(0, len(self.sets)-1) if (np.array_equal(self.sets[i][:-1], self.sets[i+1][:-1]) and not self.sets[i][-1] == self.sets[i+1][-1])]
    def merge(self, clash,ep=-1,root_id=-1):
        root_id_compare = str(root_id)
        m = np.array(clash)
        weight_dict = {}
        loc_dict = {}

        for merge in m:
            merge_key = tuple(sorted(tuple(merge[:2])))
            try:
                weight_dict[merge_key] += 1
            except:
                weight_dict[merge_key] = 1

            try:
                loc_dict[merge_key] += [tuple(merge[2:])]
            except:
                loc_dict[merge_key] = [tuple(merge[2:])]
        merge_df = pd.DataFrame()
        for c in weight_dict:
            presyn = str(c[0])
            postsyn = str(c[1])
            to_merge = -1
            if presyn==root_id_compare:
                to_merge=postsyn
                # print("PRE", to_merge, root_id)
            elif postsyn==root_id_compare:
                to_merge=presyn
                # print("POST", to_merge, root_id)

            else:
                print(root_id, presyn, postsyn)
                continue


            weight = weight_dict[c]
            locs = loc_dict[c]
            merge_df = merge_df.append({"EP":ep, "root_id":root_id_compare, "seg_id":to_merge, "Weight":weight, "Merge Locations":locs},ignore_index=True)
        return merge_df
    
def merge_paths(path_list,rids,ep,root_id):
    inter = Intersection(path_list,rids)
    inter.concArrays()
    inter.sortList()
    clash =  inter.clash()
    weighted_merge = inter.merge(clash,ep,root_id)
    return weighted_merge

def find_center(seg, seg_id):
    segmentation_coords = np.array(np.argwhere(seg == seg_id))
    return np.mean(segmentation_coords, axis=1)

def get_contacts(seg, root_id):
        from scipy.ndimage.filters import convolve
        import copy
        import agents.data_loader as data_loader

        # seg = np.squeeze(data_loader.get_seg(*bound))

        seg_mask = copy.deepcopy(seg)
        seg_copy = copy.deepcopy(seg)

        seg_mask[np.logical_not(seg_mask == root_id)] = 0
        seg_mask[seg_mask > 0] = 1
        kernel = np.ones((3,3,3), np.uint8)
        seg_dilate = convolve(seg_mask, kernel, mode="constant", cval=0)
        # seg_copy[seg_dilate > 0] = 0

        close_ids = np.unique(seg_copy[seg_dilate > 0])

        centers = {}
        sizes = {}

        points = np.argwhere(seg_copy == root_id)
        points_z = points[:,2]
        centers_list = []
        for z in range(np.min(points_z), np.max(points_z)+1):
            points_slice = np.argwhere(seg_copy[:,:,z] == root_id)
            centers_list.append([*list(np.mean(points_slice, axis=0).astype(int)), z])
        centers[root_id] = centers_list
        print(centers[root_id])
        sizes[root_id] = np.sum(seg_copy == root_id)

        for rid in close_ids:
            points = np.argwhere(seg_copy == rid)
            points_z = points[:,2]
            centers_list = []
            for z in range(np.min(points_z), np.max(points_z)+1):
                points_slice = np.argwhere(seg_copy[:,:,z] == rid)
                centers_list.append([*list(np.mean(points_slice, axis=0).astype(int)), z])
            centers[rid] = centers_list
            sizes[rid] = len(points)

        return close_ids, centers, sizes

def create_post_matrix(pos_histories, data_shape):
    pos_matrix = np.full(data_shape, np.nan)
    for i, h in enumerate(pos_histories):
        for p in h:
            p = p.astype(int)
            try:
                pos_matrix[p[0], p[1],p[2]] = i
            except:
                pass
    return pos_matrix


import matplotlib.pyplot as plt
import numpy as np


def remove_keymap_conflicts(new_keys_set):
    """
    Removes matplotlib in-built keybinding conflicts
    :param new_keys_set: dict of chars to be unbound
    :return:
    """
    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(membranes, positions=None, start_slice=-1):
    """
    View 3d volume one slice at a time
    :param membranes: 3D numpy array of greyscale data
    :param positions: [Optional] Pointcloud of agent positions
    :return:
    """
    if positions is None:
        i = np.zeros_like(membranes)

    remove_keymap_conflicts({"j", "k"})
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.paths = positions
    ax.mems = membranes

    if start_slice == -1:
        ax.index = membranes.shape[2] // 2
    else:
        ax.index = start_slice
    # make the colormaps
    mem_colors = plt.get_cmap("Greys")
    agent_colors = plt.get_cmap("tab20")

    # Initialize color maps to eliminate background color
    agent_colors._init()

    # create your alpha array and fill the colormap with them.
    agent_colors._lut[0, -1] = 0
    ax.imshow(
        membranes[:, :, ax.index].T, interpolation="nearest", cmap=mem_colors, origin="upper"
    )
    ax.imshow(
        positions[:, :, ax.index].T, interpolation="nearest", cmap=agent_colors, origin="upper"
    )

    ax.set_title(start_slice)
    fig.canvas.mpl_connect("key_press_event", process_key)
    # fig.canvas.mpl_connect('scroll_event', process_key)

def process_key(event):
    """
    Action to perform once the 'up' or 'down' key is pressed
    :param event: action of pushing the key, processed by fig.canvas.mpl_connect()
    :return:
    """
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == "j":
        move_slice(ax, -1)

    elif event.key == "k":
        move_slice(ax, 1)

    if event.button == "up":
        move_slice(ax, 1)
    elif event.button == "down":
        move_slice(ax, -1)

    fig.canvas.draw()


def move_slice(ax, sign):
    """
    Go to the next or previous slice
    :param ax: Which axis of plt object to move
    :param sign: Which direction to move the volume
    :return:
    """
    ax.index = (ax.index + sign) % ax.paths.shape[2]  # wrap around using %
    ax.images[0].set_array(ax.mems[:, :, ax.index].T)
    ax.images[1].set_array(ax.paths[:, :, ax.index].T)
    ax.set_title(ax.index)

def get_soma(root_id:str,cave_client, num=True):

    soma = cave_client.materialize.query_table(
        "nucleus_neuron_svm",
        filter_equal_dict={'pt_root_id':root_id},
        materialization_version = 117
    )
    if num:
        return soma.shape[0]
    else:
        return soma

def get_syn_counts(root_id:str, cave_client):
    pre_synapses = cave_client.materialize.query_table(
        "synapses_pni_2", 
        filter_in_dict={"pre_pt_root_id": [root_id]},
        select_columns=['ctr_pt_position', 'pre_pt_root_id']
    )

    post_synapses = cave_client.materialize.query_table(
        "synapses_pni_2", 
        filter_in_dict={"post_pt_root_id": [root_id]},
        select_columns=['ctr_pt_position', 'post_pt_root_id']
    )

    return len(pre_synapses), len(post_synapses)