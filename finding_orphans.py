import sys
from collections.abc import Iterable
from agents import data_loader
from caveclient import CAVEclient
from cloudvolume import CloudVolume, VolumeCutout
import numpy as np
from tqdm import tqdm
from orphan_extension.utils.cast_to_bounds import cast_points_within_bounds


class OrphanError(Exception):
    pass


class Orphans:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    # Gets all the seg ids within a given subvolume and organizes by size of process
    def get_unique_seg_ids_em(self) -> list:

        # Get entire EM data - uncomment after testing
        # em = CloudVolume('s3://bossdb-open-data/iarpa_microns/minnie/minnie65/seg', use_https=True, mip=0, parallel=True, fill_missing=True, progress=True)

        # Get seg ids in the specified subvolume
        seg_ids_sv = data_loader.get_seg(self.x_min, self.x_max, self.y_min,
                                         self.y_max, self.z_min, self.z_max)

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
        unique_seg_ids_sv = unique_seg_ids_sv[unique_seg_ids_sv != None]

        # Organizing seg ids in subvolume by size
        seg_ids_by_size = {}
        for seg_id in (pbar := tqdm(unique_seg_ids_sv)):
            pbar.set_description('Organizing seg_ids by size')
            # seg_ids_by_size[seg_id] = int(em[em == seg_id].sum()) # Uncomment after testing to organize seg ids by size considering whole data
            seg_ids_by_size[seg_id] = int(
                seg_ids_sv[seg_ids_sv == seg_id].sum())

        seg_ids_by_size = sorted(seg_ids_by_size.items(),
                                 key=lambda x: x[1], reverse=True)
        return seg_ids_by_size  # Sorted in descending order

    # Get the list of orphans within a given subvolume organized by largest orphan in subvolume first

    def get_orphans(self) -> dict:
        unique_seg_ids = self.get_unique_seg_ids_em()

        # Getting all the orphans
        orphans = {}

        for seg_id_and_size in (pbar := tqdm(unique_seg_ids)):
            pbar.set_description('Finding orphans')
            seg_id = seg_id_and_size[0]
            if (data_loader.get_num_soma(str(seg_id)) == 0):
                orphans[seg_id] = [seg_id_and_size[1]]

        return orphans  # list of seg_ids that are orphans in given subvolume

    # Input: processes is a dictionary with key = seg_id, value = list of attributes
    # Returns: updated processes so that value also includes the type of the process
    def get_process_type(self, processes: dict) -> dict:
        for seg_id, attributes in (pbar := tqdm(processes.items())):
            pbar.set_description('Finding process type')
            num_pre_synapses, num_post_synapses = data_loader.get_syn_counts(
                str(seg_id))
            if (num_pre_synapses > num_post_synapses):
                attributes.append('axon')
            elif (num_post_synapses > num_pre_synapses):
                attributes.append('dendrite')
            else:
                attributes.append('unconfirmed')

        return processes


def bounding_box_coords(point: Iterable, boxdim: Iterable = [100, 100, 100]) -> list:
    # Data bounds not validated
    data_bounds = [26000,220608,30304,161376,14825,27881]
    
    # Confirm that entry is 3dim
    if len(point) != 3:
        raise OrphanError(
            "Point passed to func bounding_box_coords() must be an iterable of length 3.")
    if len(boxdim) != 3:
        raise OrphanError("Box dimensions passed to func bounding_box_coords() must be 3 dimensional")
    
    # Check bound validity and cast to new bounds
    casted_bounds = cast_points_within_bounds(point, data_bounds, boxdim)
    
    return casted_bounds


def get_pot_extension(self, endpoint_coords):

    if (len(endpoint_coords) != 3):
        # FIX THIS - SHOULD BE DIFF TYPE OF ERROR
        raise OrphanError(
            "get_pot_extension needs all 3 coordinates of endpoint to extend!")

    # Get the coordinates of the bounding box around the endpoint
    endpoint_bounding_box_coords = self.bounding_box_coords(endpoint_coords)

    # Get a preliminary list of all seg ids within bounding box
    pot_ex = self.get_unique_seg_ids_em(endpoint_bounding_box_coords)

    # Get seg id of current fragment
    process_seg_id = data_loader.get_seg(
        endpoint_coords[0], endpoint_coords[0], endpoint_coords[1], endpoint_coords[1], endpoint_coords[2], endpoint_coords[2])

    # Remove current seg id from the list of potential extensions
    pot_ex = pot_ex[str(pot_ex) != str(process_seg_id)]

    # Get type of current process and all other processes
    curr_process_type = {process_seg_id: []}
    self.get_process_type(curr_process_type)

    curr_process_type = curr_process_type[1][0]

    self.get_process_type(pot_ex)

    # Filter out all other processes whose type!= current process type
    pot_ex = self.remove_diff_types(curr_process_type, pot_ex)

    return pot_ex  # Return all potential extensions after removing confirmed other types


def remove_diff_types(process_type, pot_ex):
    for seg_id, attributes in pot_ex.items():
        if (process_type not in attributes or "unconfirmed" not in attributes):
            del pot_ex[seg_id]

    return pot_ex    


if __name__ == "__main__":
    x_min = 115167
    x_max = 116414
    y_min = 91739
    y_max = 93796
    z_min = 21305
    z_max = 21315

    bounds = bounding_box_coords([115267, 91839, 21405])
    print(bounds)
    orphanclass = Orphans(*bounds)
    orphans = orphanclass.get_orphans()
    print("Number of orphans:", len(orphans))
    orphanclass.get_process_type(orphans)
    print("Orphans", orphans)
