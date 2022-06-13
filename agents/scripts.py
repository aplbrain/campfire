import backoff
import numpy as np
from scipy.ndimage.filters import convolve
import pandas as pd
from caveclient import CAVEclient
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import backoff

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
def create_queue(data_shape, n_pts, sampling_type="lin",
    isotropy=10, segmentation=np.zeros(1), root_id=-1, endpoint_nm=None):
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
        return q
    if sampling_type == "extension":
        q = []
        ids, all_ids, centers, sizes_list = get_contacts(segmentation, root_id, conv_size=6)
        data_size = data_shape[0] * data_shape[1]
        ct=0
        ids = ids[ids != 0]
        soma_table = get_soma(all_ids)
        polarity_table = is_dendrite(endpoint_nm, all_ids)
        for i in ids:
            if i != root_id and soma_table.get_soma(i) > 0:
                continue
            if i != root_id and not polarity_table.dendrite_filt(root_id, i):
                continue
            centers_list = centers[i]
            sizes_ext = sizes_list[i]
            for j in range(len(centers_list)):
                n_gen = int(sizes_ext[j] / data_size * n_pts)
                if sizes_ext[j] > 0:
                    n_gen += 1

                for _ in range(n_gen):
                    q.append([*centers_list[j]])
                ct += 1
        return q, soma_table, polarity_table
    if sampling_type == "extension_only":
        q = []
        centers_list, sizes_list = get_seg_slices(root_id, segmentation)
        polarity_table = is_dendrite(endpoint_nm, root_id)
        polarity = polarity_table.dendrite_df[polarity_table.dendrite_df.seg_id==root_id].iloc[0].polarity
        data_size = data_shape[0] * data_shape[1]
        ct=0
        for j in range(len(centers_list)):
            n_gen = int(sizes_list[j] / data_size * n_pts)
            if sizes_list[j] > 0:
                n_gen += 1
            for _ in range(n_gen):
                q.append([*centers_list[j]])
            ct += 1
        return q, polarity

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

def get_contacts(seg, root_id,conv_size=3):
    from scipy.ndimage.filters import convolve
    import copy
    import agents.data_loader as data_loader

    seg_mask = copy.deepcopy(seg)
    seg_copy = copy.deepcopy(seg)

    seg_mask[np.logical_not(seg_mask == root_id)] = 0
    seg_mask[seg_mask > 0] = 1
    kernel = np.ones((conv_size,conv_size,conv_size), np.uint8)
    seg_dilate = convolve(seg_mask, kernel, mode="constant", cval=0)

    close_ids = np.unique(seg_copy[seg_dilate > 0])
    all_ids = np.unique(seg_copy)

    centers = {}
    sizes = {}

    points = np.argwhere(seg_copy == root_id)
    points_z = points[:,2]
    centers_list = []
    sizes_list = []
    for z in range(np.min(points_z), np.max(points_z)+1):
        if z < 2 or z > 198:
            continue
        points_slice = np.argwhere(seg_copy[:,:,z] == root_id)
        centers_list.append([*list(np.mean(points_slice, axis=0).astype(int)), z])
        sizes_list.append(len(points_slice))
    centers[root_id] = centers_list
    sizes[root_id] = sizes_list
    for rid in close_ids:
        points = np.argwhere(seg_copy == rid)
        points_z = points[:,2]
        centers_list = []
        sizes_list = []
        for z in range(np.min(points_z), np.max(points_z)+1):
            points_slice = np.argwhere(seg_copy[:,:,z] == rid)
            mean_pt = np.mean(points_slice, axis=0)
            if points_slice.shape[0] > 0 and seg[int(mean_pt[0]), int(mean_pt[1]), z] == rid:
                centers_list.append([*list(mean_pt.astype(np.uint64)), z])
                sizes_list.append(len(points_slice))

        centers[rid] = centers_list
        sizes[rid] = sizes_list

    return close_ids, all_ids, centers, sizes

def get_seg_slices(root_id, seg):
    import  copy
    seg_mask = copy.deepcopy(seg)

    seg_mask[np.logical_not(seg_mask == root_id)] = 0
    seg_mask[seg_mask > 0] = 1
    sizes_list = []

    points = np.argwhere(seg == root_id)
    points_z = points[:,2]
    centers_list = []
    for z in range(np.min(points_z), np.max(points_z)+1):
        if z < 2 or z > 198:
            continue
        points_slice = np.argwhere(seg[:,:,z] == root_id)
        centers_list.append([*list(np.mean(points_slice, axis=0).astype(int)), z])
        sizes_list.append(len(points_slice))

    return centers_list, sizes_list

def position_merge(ep, root_id, merges, endpoint_nm):
    merge_df = pd.DataFrame()

    ms = set()
    for m in merges:
        ms.add(m[0])
        ms.add(m[1])

    m_list = list(ms)
    soma_table = get_soma(m_list)
    polarity_table = is_dendrite(endpoint_nm, m_list)
    
    for i, k in enumerate(merges):
        if k[0] == root_id:
            extension = k[1]
            weight = merges[k]
        elif k[1] == root_id:
            extension = k[0]
            weight = merges[k]
        else:
            continue
        if extension == 0:
            continue
        if soma_table.get_soma(extension) > 0:
            continue
        if polarity_table.dendrite_filt(root_id, extension):
            continue
        d = {"EP":[ep], "Root_id":str(root_id), "Extension":str(extension), "Weight":weight}

        merge_df = pd.concat([merge_df, pd.DataFrame(d, index=[i])])
    return merge_df

def find_center(seg, seg_id):
    segmentation_coords = np.array(np.argwhere(seg == seg_id))
    return np.mean(segmentation_coords, axis=1)

def create_post_matrix(pos_histories, seg, data_shape,merge=False):
    pos_matrix = np.full(data_shape, np.nan)
    merge_d = {}
    for i, h in enumerate(pos_histories):
        p0 = h[0]
        p0 = p0.astype(int)
        init_rid = seg[p0[0], p0[1], p0[2]]
        added = set()
        for p in h:
            p = p.astype(int)
            seg_id = seg[p[0], p[1], p[2]]
            if merge and seg_id != init_rid:
                if seg_id not in added:
                    try:
                        merge_d[(init_rid, seg_id)] += 1
                    except:
                        merge_d[(init_rid, seg_id)] = 1
                    added.add(seg_id)
            pos_matrix[p[0], p[1],p[2]] = init_rid
    return pos_matrix, merge_d

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

class get_soma():
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def __init__(self, seg_ids):
        self.seg_ids = seg_ids
        cave_client = CAVEclient('minnie65_phase3_v1')
        soma = cave_client.materialize.query_table(
            "nucleus_neuron_svm",
            materialization_version = 117,
            filter_in_dict={'pt_root_id':seg_ids},
            select_columns=['id','pt_root_id', 'pt_position']
        )
        soma_dict = {}
        for s in seg_ids:
            soma_dict[s] = soma[soma['pt_root_id'] == s].shape[0]
        self.soma_dict = soma_dict

    def get_soma(self, seg_id):
        return self.soma_dict[seg_id]

def make_bounding_box(point, distance):

    point_a = [coord - distance for coord in point]
    point_b = [coord + distance for coord in point]

    return [point_a, point_b]

class is_dendrite():
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def __init__(self, endpoint, seg_ids):
        new_ids, seg_dict = get_current_seg_ids(seg_ids)
        # Create a client for the Minnie65 PCG and Tables
        client = CAVEclient('minnie65_phase3_v1')
        bounding_box = make_bounding_box(endpoint, 10000)
        synapse_table = 'synapses_pni_2'
        post_ids = client.materialize.query_table(synapse_table,
        filter_in_dict={'post_pt_root_id': new_ids},
        filter_spatial_dict={'post_pt_position': bounding_box},
        select_columns=['post_pt_root_id','post_pt_position']
        )

        pre_ids = client.materialize.query_table(synapse_table,
        filter_in_dict={'pre_pt_root_id': new_ids},
        filter_spatial_dict={'pre_pt_position': bounding_box},
        select_columns=['pre_pt_root_id','pre_pt_position']
        )
        dendrite_df = pd.DataFrame()
        for ind, i in enumerate(new_ids):
            len_post = len(post_ids[post_ids["post_pt_root_id"] == i])
            len_pre = len(pre_ids[pre_ids["pre_pt_root_id"] == i])
            if len_post == len_pre and len_post == 0:
                polarity="No synapses"
                mixed_polarity=False
            elif len_post > 1 and len_pre > 0:
                polarity="Dendrite"
                mixed_polarity=True
            elif len_post > 1 and len_pre == 0:
                polarity="Denrite"
                mixed_polarity=False
            else:
                polarity="Axon"
                mixed_polarity=False
            dendrite_df = pd.concat([dendrite_df, pd.DataFrame({'seg_id':seg_dict[i], 'polarity':polarity, 'mixed_polarity':mixed_polarity}, index=[ind])])
        self.dendrite_df = dendrite_df

    def dendrite_filt(self, root_id, seg_id):
        polarity_root = self.dendrite_df[self.dendrite_df.seg_id==root_id].polarity.iloc[0]
        seg = self.dendrite_df[self.dendrite_df.seg_id==seg_id].iloc[0]
        polarity_seg = seg.polarity
        mixed_polarity = seg.mixed_polarity
        if polarity_root == polarity_seg:
            return False
        if polarity_root == "Unknown" or polarity_seg == "Unknown":
            return True
        else:
            return True

def shift_detect(em, radius, n_windows, n_hits=-1,threshold=10,zero_threshold=.75):
    if n_hits == -1:
        if n_windows > 1:
            n_hits = n_windows - 1
        else:
            n_hits=1
    em_shape = em.shape
    centers = np.linspace(0,em_shape[0],n_windows + 2).astype(int)[1:-1]
    errors = np.full(em_shape[2], True)
    errors_zero = np.full(em_shape[2], False)

    if np.sum(em[:,:,0]==0) > zero_threshold*em_shape[0]*em_shape[1]:
        errors_zero[0] = True
    for i in range(1,em.shape[2]):
        means = []
        locs = []
        last_frame = em[:,:,i-1]
        frame=em[:,:,i]
        if np.sum(em[:,:,i]==0) > zero_threshold*em_shape[0]*em_shape[1]:
            errors_zero[i] = True
        for k in range(n_windows):
            if centers[k]-radius < 0 or centers[k]+radius > em_shape[0]:
                continue
            xmin, xmax = centers[k]-radius, centers[k]+radius
            
            last_frame_center = last_frame[xmin:xmax, xmin:xmax]
            res = cv2.matchTemplate(frame, last_frame_center, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            mse = np.sqrt(np.sum(np.square(np.array(min_loc)-np.array((centers[k]-radius,centers[k]-radius)))))
            means.append(mse)
            locs.append(min_loc)
        if np.sum(np.array(means) < threshold) >= n_hits:
            errors[i-1] = False
            errors[i] = False
    return errors, errors_zero

def remove_shifts(em, errors,zero_or_remove='zero'):
    if zero_or_remove == 'zero':
        em[:,:,errors] = 0
    elif zero_or_remove=='remove':
        em = em[:,:,errors]
    return em

def alignImages(im1, im2, MAX_FEATURES=500, GOOD_MATCH_PERCENT = 0.15):

  # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    try:
        matches = list(matcher.match(descriptors1, descriptors2, None))
    except:
        return("Fail Blank", np.zeros((3,3)))
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
#     cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    try:
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    except:
        return("Fail Misaligned", np.zeros((3,3)))
    # Use homography
    height, width  = im2.shape
    try:
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
    except:
        return("Fail Misaligned", np.zeros((3,3)))
    return im1Reg, h

def get_public_seg_ids(seg_ids):
    if type(seg_ids) != list:
        seg_ids = [seg_ids]
    client = CAVEclient("minnie65_phase3_v1")
    ts = client.materialize.get_timestamp(117)
    past_ids = client.chunkedgraph.get_past_ids(root_ids=seg_ids,timestamp_past=ts)
    ids = list(past_ids['past_id_map'].keys())
    public_ids = []
    for j, id_key in enumerate(ids):
        k = past_ids['past_id_map'][id_key]
        public_ids.append(int(k[0]))

    return public_ids

def get_current_seg_ids(seg_ids):
    if type(seg_ids) != list:
        try:
            seg_ids = list(seg_ids)
        except TypeError:
            seg_ids = [seg_ids]
    client = CAVEclient("minnie65_phase3_v1")
    out_of_date_mask = client.chunkedgraph.is_latest_roots(seg_ids)
    seg_ids = np.array(seg_ids)
    out_of_date_seg_ids = seg_ids[np.logical_not(out_of_date_mask)]
    seg_dict={}
    for s in out_of_date_seg_ids:
        new_s = client.chunkedgraph.get_latest_roots(s)
        seg_ids[seg_ids == s] = new_s[-1]
        seg_dict[new_s[-1]]=s
    for s in seg_ids[out_of_date_mask]:
        seg_dict[s] = s
    return seg_ids, seg_dict

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
        return [[self.sets[i][-1], self.sets[i+1][-1], *self.sets[i][:-1]] for i in range(0, len(self.sets)-1) if (np.array_equal(self.sets[i][:-1], self.sets[i+1][:-1]) and not self.sets[i][-1] == self.sets[i+1][-1])]

    def merge(self, clash, soma, polarity, ep=-1,root_id=-1):
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
            elif postsyn==root_id_compare:
                to_merge=presyn
            else:
                continue
            if soma.get_soma(int(to_merge)) > 0:
                continue
            if polarity.dendrite_filt(int(root_id), int(to_merge)):
                continue
            weight = weight_dict[c]
            locs = loc_dict[c]
            merge_df = merge_df.append({"EP":ep, "root_id":root_id_compare, "seg_id":to_merge, "Weight":weight, "Merge Locations":locs},ignore_index=True)
        return merge_df
    
def merge_paths(path_list,rids,ep,root_id, soma, polarity):
    inter = Intersection(path_list,rids)
    try:
        inter.concArrays()
    except:
        return pd.DataFrame()
    inter.sortList()
    clash =  inter.clash()
    weighted_merge = inter.merge(clash,soma, polarity, ep,root_id)
    return weighted_merge