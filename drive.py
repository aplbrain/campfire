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
import pandas as pd
import aws.sqs as sqs
import sys

def run_endpoints(root_id,radius=(100,100,10), resolution=(8,8,40), unet_bound_mult=2, return_opacity_seg=False,save_format='pickle'):
    # SPINE FINDING
    import pickle

    root_merge_df = pd.DataFrame()
    big_merge_df = pd.DataFrame()
    client = CAVEclient('minnie65_phase3_v1')
    end_points = spine_finding.find_endpoints(root_id, 
                               client=client,
                               refine='all',
                               root_point_resolution=[4, 4, 40],
                               collapse_soma=True,
                               n_parallel=8)
    end_points = pickle.load(open('864691135761488438_endpoints.p','rb'))
    if save_format == 'sqs':
        queue_url = sqs.get_or_create_queue('Endpoints')
        entries=sqs.construct_endpoint_entries(end_points, root_id)
        while(len(entries) > 0):
            entries_send = entries[:10]
            entries = entries[10:]
            sqs.send_batch(queue_url, entries_send)
    elif save_format == 'pickle':
        import pickle
        pickle.dump(end_points, open(f"./data/{root_id}_endpoints.p", "wb"))


def drive(root_id, radius, resolution, unet_bound_mult, ep='sqs', save_pd=True,device='cpu'):
        if ep == 'sqs':
            eq = sqs.get_job_from_queue('Endpoints')
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

        seg = np.squeeze(data_loader.get_seg(*bound))

        vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
        em = np.squeeze(vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3], bound_EM[4]:bound_EM[5]])
        mem_seg = membranes.segment_membranes(em, device_s=device)
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
        if merges_root.iloc[0].M1 == root_id:
            merges_root.rename({"M1":"root_id", "M2":"segment_id"})
        else:
            merges_root.rename({"M2":"root_id", "M1":"segment_id"})

        if save_pd:
            merges.to_csv(f"./merges_{root_id}_{endpoint}.csv")
            merges_root.to_csv(f"./merges_root_{root_id}_{endpoint}.csv")

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

        opacity_merges[seg == seg_id] = merges_root[(merges_root.M1 == seg_id) | (merges_root.M2 == seg_id)]["Weight"] / sum_weight
        seg_merge_output[seg == seg_id] = i
    seg_merge_output[seg_merge_output==0]=np.nan
    return opacity_merges, seg_merge_output

    # Iterate through endpoints, running agents and merging
    for ep in end_points:
        merges, merges_root, pos_matrix, seg = drive(ep, root_id, radius, resolution,unet_bound_mult)
        if return_opacity_seg:
            opacity_merges, seg_merge_output = visualize_merges(merges_root, seg, root_id)
            np.savez(f"./opacity_seg_{root_id}_{ep}.npz", opacity_merges,seg_merge_output)
        # Merges are indexed by their endpoint
        big_merge_df = big_merge_df.append(merges)
        root_merge_df = root_merge_df.append(merges_root)
    if save_format == 'pickle':
        big_merge_df.to_pickle(f"./merges_{root_id}.p")
        root_merge_df.to_pickle(f"./merges_root_{root_id}.p")
    elif save_format == 'csv':
        merges.to_csv(f"./merges_{root_id}_{ep}.csv")
        merges_root.to_csv(f"./merges_root_{root_id}_{ep}.csv")

if __name__ == "__main__":
    mode = sys.argv[1]
    root_id = sys.argv[2]
    if mode == 'endpoints':
        print()
        run_endpoints(root_id, **dict(arg.split('=') for arg in sys.argv[3:]))