import sys
from agents import data_loader
from caveclient import CAVEclient
from cloudvolume import CloudVolume, VolumeCutout
import numpy as np
from tqdm import tqdm


# Gets all the seg ids within a given subvolume and organizes by size of process
def get_unique_seg_ids_em(x_min, x_max, y_min, y_max, z_min, z_max):

    # Get entire EM data - uncomment after testing
    # em = CloudVolume('s3://bossdb-open-data/iarpa_microns/minnie/minnie65/seg', use_https=True, mip=0, parallel=True, fill_missing=True, progress=True)

    # Get seg ids in the specified subvolume
    seg_ids_sv = data_loader.get_seg(x_min, x_max, y_min, y_max, z_min, z_max)

    # Get rid of the 4th dimension since its magnitude is 1
    seg_ids_sv = np.squeeze(seg_ids_sv)

    # List of all unique seg ids in the 3d subvolume
    unique_seg_ids_sv = np.unique(seg_ids_sv)

    # Removing the first element (artifacts) of unique_seg_ids_sv
    """
    Drop all zeros in array instead of only first, handles edge cases
    unique_seg_ids_sv = np.delete(unique_seg_ids_sv, 0)
    """

    unique_seg_ids_sv = unique_seg_ids_sv[unique_seg_ids_sv != 0]

    # Organizing seg ids in subvolume by size
    seg_ids_by_size = {}
    for seg_id in tqdm(unique_seg_ids_sv):
        # seg_ids_by_size[seg_id] = int(em[em == seg_id].sum()) # Uncomment after testing to organize seg ids by size considering whole data
        seg_ids_by_size[seg_id] = int(seg_ids_sv[seg_ids_sv == seg_id].sum())

    seg_ids_by_size = sorted(seg_ids_by_size.items(),
                             key=lambda x: x[1], reverse=True)
    return seg_ids_by_size  # Sorted in descending order


# Get the list of orphans within a given subvolume organized by largest orphan in subvolume first
def get_orphans(x_min, x_max, y_min, y_max, z_min, z_max):
    unique_seg_ids = get_unique_seg_ids_em(
        x_min, x_max, y_min, y_max, z_min, z_max)

    # Getting all the orphans
    orphans = {}

    for seg_id_and_size in tqdm(unique_seg_ids):
        seg_id = seg_id_and_size[0]
        if (data_loader.get_num_soma(str(seg_id)) == 0):
            orphans[seg_id] = [seg_id_and_size[1]]

    return orphans  # list of seg_ids that are orphans in given subvolume


# Input: processes is a dictionary with key = seg_id, value = list of attributes
# Returns: updated processes so that value also includes the type of the process
def get_process_type(processes):
    for seg_id, attributes in processes.items():
        num_pre_synapses, num_post_synapses = data_loader.get_syn_counts(
            str(seg_id))
        if (num_pre_synapses > num_post_synapses):
            attributes.append('axon')
        elif (num_post_synapses > num_pre_synapses):
            attributes.append('dendrite')
        else:
            attributes.append('unconfirmed')
    return processes


if __name__ == "__main__":
    x_min = 115167
    x_max = 116414
    y_min = 91739
    y_max = 93796
    z_min = 21305
    z_max = 21315
    orphans = get_orphans(x_min, x_max, y_min, y_max, z_min, z_max)
    print("Number of orphans:", len(orphans))
    print(type(orphans))
    get_process_type(orphans)
    print("Orphans", orphans)
