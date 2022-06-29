import sys
from collections.abc import Iterable
from agents import data_loader
from caveclient import CAVEclient
from cloudvolume import CloudVolume, VolumeCutout
import numpy as np
from tqdm import tqdm
from orphan_extension.utils.cast_to_bounds import cast_points_within_bounds
from orphan_extension.utils.multi_loader import multi_proc_type, multi_soma_count
from tip_finding import tip_finding


class OrphanError(Exception):
    pass


class Orphans:
    def __init__(self, bounds_list: list):
        self.bounds_list = bounds_list
    
    # def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
    #     self.x_min = x_min
    #     self.x_max = x_max
    #     self.y_min = y_min
    #     self.y_max = y_max
    #     self.z_min = z_min
    #     self.z_max = z_max

    
    def get_unique_seg_ids_em(self, coords=None) -> list:
        """
        Gets all the seg ids within a given subvolume and organizes by size of process. Returns list of tuples: (seg_id, size)
        """
        if (coords != None and len(coords) != 6):  # CHANGE THE ERROR THROWN!
            raise OrphanError("get_unique_seg_ids_em needs 6 coordinates!!")
        
        x_min, x_max, y_min, y_max, z_min, z_max = coords if coords else self.bounds_list

        seg_ids_sv = data_loader.get_seg(x_min, x_max, y_min,
                                        y_max, z_min, z_max)
        # Get entire EM data - uncomment after testing
        # em = CloudVolume('s3://bossdb-open-data/iarpa_microns/minnie/minnie65/seg', use_https=True, mip=0, parallel=True, fill_missing=True, progress=True)

        # Get seg ids in the specified subvolume

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
            seg_ids_by_size[seg_id] = [int(
                np.sum(seg_ids_sv == seg_id))]

        seg_ids_by_size = sorted(seg_ids_by_size.items(),
                                 key=lambda x: x[1], reverse=True)
        # Returns a list of tuples with first element as seg id, second elmenet of tuple is a list containing size
        return seg_ids_by_size  # Sorted in descending order


    def get_orphans(self, coords=None) -> dict:
        """
        Get the list of orphans within a given subvolume organized by largest orphan in subvolume first
        """
        unique_seg_ids = self.get_unique_seg_ids_em(coords)
        unique_seg_ids_l = [i[0] for i in unique_seg_ids]

        # Getting all the orphans
        orphans = multi_soma_count(unique_seg_ids_l)
        orphans = list({k:v for k, v in orphans.items() if v==0}.keys())

        orphan_dict = {}

        for seg_id_and_size in (pbar := tqdm(unique_seg_ids)):
            pbar.set_description('Labeling orphans')
            if seg_id_and_size[0] in orphans:
                orphan_dict[seg_id_and_size[0]] = seg_id_and_size[1]

        return orphan_dict  # dict of seg_ids that are orphans in given subvolume


    def get_process_type(self, processes: dict) -> dict:
        """
        Input: processes is a dictionary with key = seg_id, value = list of attributes
        Returns: updated processes so that value also includes the type of the process
        """
        proc_dict = multi_proc_type(list(processes.keys()))
        for seg_id, attributes in (pbar := tqdm(processes.items())):
            pbar.set_description('Adding process type')
            attributes.append(proc_dict[seg_id])

        return processes


    def get_pot_extensions(self, endpoint_coords):

        if (len(endpoint_coords) != 3):
            # FIX THIS - SHOULD BE DIFF TYPE OF ERROR
            raise OrphanError(
                "get_pot_extension needs all 3 coordinates of endpoint to extend!")

        # Get the coordinates of the bounding box around the endpoint
        endpoint_bounding_box_coords = bounding_box_coords(endpoint_coords)

        # Get a preliminary list of all seg ids within bounding box
        # pot_ex = self.get_unique_seg_ids_em(*endpoint_bounding_box_coords)
        pot_ex = self.get_unique_seg_ids_em(endpoint_bounding_box_coords)

        # Get seg id of current fragment
        curr_process_seg_id = data_loader.get_seg(
            endpoint_coords[0], endpoint_coords[0], endpoint_coords[1], endpoint_coords[1], endpoint_coords[2], endpoint_coords[2])

        # Get type of all processes
        pot_ex = dict(pot_ex)
        pot_ex = self.get_process_type(pot_ex)

        # Get type of current process
        curr_process_type = pot_ex[curr_process_seg_id][1]

        # Remove current seg id from the list of potential extensions - REWORK TO USE DELETE
        # pot_ex = pot_ex[str(pot_ex) != str(curr_process_seg_id)]
        del pot_ex[curr_process_seg_id]

        # Filter out all other processes whose type!= current process type
        pot_ex = remove_diff_types(curr_process_type, pot_ex)

        return pot_ex  # Return all potential extensions after removing confirmed other types


def bounding_box_coords(point: Iterable, boxrad: Iterable = [100, 100, 10]) -> list:
    # Data bounds not validated
    data_bounds = [26000, 220608, 30304, 161376, 14825, 27881]

    # Confirm that entry is 3dim
    if len(point) != 3:
        raise OrphanError(
            "Point passed to func bounding_box_coords() must be an iterable of length 3.")
    if len(boxrad) != 3:
        raise OrphanError(
            "Box dimensions passed to func bounding_box_coords() must be 3 dimensional")

    # Check bound validity and cast to new bounds
    casted_bounds = cast_points_within_bounds(point, data_bounds, boxrad)

    return casted_bounds


def remove_diff_types(process_type, pot_ex):
    for seg_id, attributes in pot_ex.items():
        if (process_type not in attributes or "unconfirmed" not in attributes):
            del pot_ex[seg_id]

    return pot_ex


if __name__ == "__main__":
    # x_min = 115167
    # x_max = 116414
    # y_min = 91739
    # y_max = 93796
    # z_min = 21305
    # z_max = 21315

    # coords = [x_min, x_max, y_min, y_max, z_min, z_max]
    # orphans = get_orphans(coords)

    bounds = bounding_box_coords([115267, 91839, 21305], boxrad = [100,100,10])
    orphanclass = Orphans(bounds)
    orphans = orphanclass.get_orphans()
    print("Number of orphans:", len(orphans))
    proc_types = orphanclass.get_process_type(orphans)
    print(proc_types)

    total_size = orphans.keys()
    total_size = list(total_size)
    # total_size = np.array(total_size)
    # print(total_size)
    # print(len(proc_types), len(total_size))

