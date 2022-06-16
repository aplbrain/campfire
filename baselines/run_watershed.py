import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import os
from cloudvolume import CloudVolume
from data_utils import FILENAME_ENCODING, load_point
from scipy import ndimage as ndi
import skimage

sys.path.append("..")
from membrane_detection import membranes


def find_extension_id(point, root_id, em_vol, seg_vol, em_dir, seg_dir, radius=(256,256,32), resolution_scale=(2,2,1)):
    point = np.divide(point, resolution_scale)
    em = load_point(point, em_vol, em_dir, radius=radius)
    mem = membranes.segment_membranes(np.squeeze(em),
                                      pth="../membrane_detection/best_metric_model_segmentation2d_dict.pth",
                                      device_s='gpu')
    dist = ndi.distance_transform_edt(mem == 0, return_distances=True, return_indices=False)
    dist = skimage.morphology.reconstruction((dist-1), dist)
    watershed_labels = skimage.segmentation.watershed(-dist, connectivity=2, watershed_line=False)
    label = watershed_labels[int(radius[0]/2), int(radius[1]/2), int(radius[2]/2)]

    seg = load_point(point, seg_vol, seg_dir, radius=radius)
    ids, counts = np.unique(seg[watershed_labels == label], return_counts=True)
    if ids[0] == 0:
        ids, counts = np.delete(ids, 0), np.delete(counts, 0)
    root_id = seg[int(radius[0]/2), int(radius[1]/2), int(radius[2]/2)]
    idx = np.where(ids == root_id)
    ids, counts = np.delete(ids, idx), np.delete(counts, idx)

    return ids[np.argmax(counts)]


def main(df, em_dir, seg_dir):
    em_vol = CloudVolume("precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em",
                         use_https=True, mip=0)
    seg_vol = CloudVolume("precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/seg",
                          use_https=True, mip=0)
    for i, row in tqdm(df.iterrows(), desc='Points', total=len(df)):
        ext_id = find_extension_id(row["EP"], int(row["Root_id"][:-1]), em_vol, seg_vol, em_dir, seg_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_file', default='expanded_gt.pkl')
    parser.add_argument('--seg-dir',default='seg')
    parser.add_argument('--em-dir',default='em')
    parser.add_argument('--base-dir',default='.')
    args = parser.parse_args()
    df = pd.read_pickle(args.evaluation_file)
    main(df, os.path.join(args.base_dir, args.em_dir), os.path.join(args.base_dir, args.seg_dir))
