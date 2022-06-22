import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
from caveclient import CAVEclient
from cloudvolume import CloudVolume
from scipy import ndimage as ndi
import skimage

sys.path.append("..")
from membrane_detection import membranes


def find_extension_id(point, root_id, em_vol, seg_vol, cave_client, radius=(256, 256, 32), resolution_scale=(2, 2, 1)):
    point = np.divide(point, resolution_scale)
    em = em_vol.download_point(point, size=radius, mip=0)
    mem = membranes.segment_membranes(np.squeeze(em),
                                      pth="../membrane_detection/best_metric_model_segmentation2d_dict.pth",
                                      device_s='gpu')
    dist = ndi.distance_transform_edt(mem == 0, return_distances=True, return_indices=False)
    dist = skimage.morphology.reconstruction((dist-1), dist)
    watershed_labels = skimage.segmentation.watershed(-dist, connectivity=2, watershed_line=False)
    label = watershed_labels[int(radius[0]/2), int(radius[1]/2), int(radius[2]/2)]

    seg = seg_vol.download_point(point, size=radius, mip=0)
    sv_ids = np.asarray(np.squeeze(seg))
    unique_sv_ids, inv = np.unique(sv_ids, return_inverse=True)
    segs = cave_client.chunkedgraph.get_roots(supervoxel_ids=unique_sv_ids)
    seg_dict = dict(zip(unique_sv_ids, segs))
    seg = np.array([seg_dict[x] for x in unique_sv_ids])[inv].reshape(sv_ids.shape)

    ids, counts = np.unique(seg[watershed_labels == label], return_counts=True)
    if ids[0] == 0:
        ids, counts = np.delete(ids, 0), np.delete(counts, 0)
    # root_id = seg[int(radius[0]/2), int(radius[1]/2), int(radius[2]/2)]
    idx = np.where(ids == root_id)
    ids, counts = np.delete(ids, idx), np.delete(counts, idx)
    idx = np.flip(np.argsort(counts))
    idx, counts = ids[idx], counts[idx]

    return ids[0] if counts.size > 0 else -1, seg[int(radius[0]/2), int(radius[1]/2), int(radius[2]/2)], ids, counts


def main(df):
    em_vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em",
                         use_https=True, mip=0, bounded=False, fill_missing=True)
    seg_vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/ws",
                          use_https=True, mip=0, bounded=False, fill_missing=True)
    cave_client = CAVEclient('minnie65_phase3_v1')
    ext_ids = []
    root_ids = []
    ids_list = []
    counts_list = []
    for i, row in tqdm(df.iterrows(), desc='Points', total=len(df)):
        ext_id, root_id, ids, counts = find_extension_id(row["EP"], int(row["Root_id"][:-1]), em_vol, seg_vol, cave_client)
        ext_ids.append(ext_id)
        root_ids.append(root_id)
        ids_list.append(ids)
        counts_list.append(counts)
    df.insert(len(df.columns), "Watershed_ext", [(str(ext_id) + '/') for ext_id in ext_ids])
    df.insert(len(df.columns), "Watershed_root_id", [(str(root_id) + '/') for root_id in root_ids])
    df.insert(len(df.columns), "Watershed_ids", ids_list)
    df.insert(len(df.columns), "Watershed_counts", counts_list)
    df.to_csv("watershed_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_file', default='expanded_gt.pkl')
    parser.add_argument('--seg-dir', default='seg')
    parser.add_argument('--em-dir', default='em')
    parser.add_argument('--base-dir', default='.')
    args = parser.parse_args()
    df_gt = pd.read_pickle(args.evaluation_file)
    main(df_gt)
