from spine_finding import spine_finding
from caveclient import CAVEclient
import numpy as np
from agents import data_loader
from cloudvolume import CloudVolume
from matplotlib import pyplot as plt
from membrane_detection import membranes
from agents.scripts import precompute_membrane_vectors, create_post_matrix, merge_paths
import agents.sensor
from agents.run import run_agents

def run_endpoints(root_id,radius=(100,100,10), resolution=(8,8,40), unet_bound_mult=2, return_opacity_seg=False):
            # SPINE FINDING
    client = CAVEclient('minnie65_phase3_v1')
    end_points = spine_finding.find_endpoints(root_id, 
                               client=client,
                               refine='all',
                               root_point_resolution=[4, 4, 40],
                               collapse_soma=True,
                               n_parallel=8)
    for ep in end_points:
        merges, merges_root, pos_matrix, seg = drive(ep, root_id, radius, resolution,unet_bound_mult)
        if return_opacity_seg:
            opacity_merges, seg_merge_output = visualize_merges(merges_root, seg, root_id)
    


def drive(ep, root_id, radius, resolution, unet_bound_mult):
        endpoint = np.divide(ep, resolution).astype('int')
        precomp_file_path = f"./precompute_{root_id}_{endpoint}"
        bound  = (endpoint[0] - radius[0], 
                    endpoint[0] + radius[0],
                    endpoint[1] - radius[1],
                    endpoint[1] + radius[1],
                    endpoint[2] - radius[2],
                    endpoint[2] + radius[2])

        bound_EM  = (endpoint[0] - unet_bound_mult*radius[0], 
                    endpoint[0] + unet_bound_mult*radius[0],
                    endpoint[1] - unet_bound_mult*radius[1],
                    endpoint[1] + unet_bound_mult*radius[1],
                    endpoint[2] - radius[2],
                    endpoint[2] + radius[2])

        #     data = np.squeeze(data_loader.get_em(*bound))
        seg = np.squeeze(data_loader.get_seg(*bound))

        vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
        em = np.squeeze(vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3], bound_EM[4]:bound_EM[5]])
        mem_seg = membranes.segment_membranes(em)
        mem_to_run = mem_seg[(unet_bound_mult-1)*radius[0]:(unet_bound_mult+1)*radius[0],
                     (unet_bound_mult-1)*radius[1]:(unet_bound_mult+1)*radius[1], :].astype(float)
        compute_vectors = precompute_membrane_vectors(mem_to_run, mem_to_run, precomp_file_path, 7)

        sensor_list = [
            (agents.sensor.BrownianMotionSensor(), .1),
            (agents.sensor.PrecomputedSensor(), .2),
            ]
    
        pos_histories, seg_ids, agent_ids=run_agents(mem=mem_to_run, sensor_list=sensor_list,
                                                    precompute_fn=precomp_file_path+'.npz',
                                                    max_vel=.9,n_steps=300,segmentation=seg, root_id=root_id)
        # find merges of agents from root_id
        pos_matrix = create_post_matrix(pos_histories, mem_to_run.shape)
        merges = merge_paths(pos_histories,seg_ids, ep)
        merges_root = merges[(merges.M1 == root_id) | (merges.M2 == root_id)]

        return merges, merges_root, pos_matrix, seg

def visualize_merges(merges_root, seg, root_id):
    mset = set(merges_root.M1)
    mset.update(set(merges_root.M2))
    sum_weight = np.sum(np.asarray(merges_root["Weight"]))
    seg = np.asarray(seg)
    opacity_merges = np.zeros_like(seg).astype(float)
    seg_merge_output = np.zeros_like(seg).astype(float)
    for i, seg_id in enumerate(mset):
        if float(seg_id) == float(root_id):
            opacity_merges[seg == seg_id] = 1
            seg_merge_output[seg == seg_id] = -1
            continue
    #     print(merges_root[(merges_root.M1 == seg_id) | (merges_root.M2 == seg_id)]["Weight"] / sum_weight)
        opacity_merges[seg == seg_id] = merges_root[(merges_root.M1 == seg_id) | (merges_root.M2 == seg_id)]["Weight"] / sum_weight
        seg_merge_output[seg == seg_id] = i
    # opacity_merges[opacity_merges==0]=np.nan
    seg_merge_output[seg_merge_output==0]=np.nan
    return opacity_merges, seg_merge_output

