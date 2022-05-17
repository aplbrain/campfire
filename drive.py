from spine_finding import spine_finding
from caveclient import CAVEclient
import numpy as np
from agents import data_loader
from cloudvolume import CloudVolume
from matplotlib import pyplot as plt
from membrane_detection import membranes
from agents.scripts import precompute_membrane_vectors, create_post_matrix, merge_paths, get_soma, get_syn_counts
import agents.sensor
from agents.run import run_agents
import pandas as pd
import aws.sqs as sqs
import sys
import time

def drive(n, radius=(100,100,10), resolution=(8,8,40), unet_bound_mult=2, ep='sqs', save='sqs',device='cpu',filter_merge=True,delete=True):
    tic=time.time()
    queue_url_endpts = sqs.get_or_create_queue('Endpoints_Test')
    vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)

    n = int(n)
    if n == -1:
        n = int(1e10)
    for _ in range(n):
        if ep == 'sqs':
            ep_msg = sqs.get_job_from_queue(queue_url_endpts)
            root_id = int(ep_msg.message_attributes['root_id']['StringValue'])
            nucleus_id = int(ep_msg.message_attributes['nucleus_id']['StringValue'])
            time_point = int(ep_msg.message_attributes['time']['StringValue'])
            ep = np.array([float(p) for p in ep_msg.body.split(',')])
            if delete:
                ep_msg.delete()
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
        tic = time.time()
        vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)

        em = np.squeeze(vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3], bound_EM[4]:bound_EM[5]])
        mem_seg = membranes.segment_membranes(em, pth="./membrane_detection/best_metric_model_segmentation2d_dict.pth", device_s=device)
        mem_to_run = mem_seg[(unet_bound_mult-1)*radius[0]:(unet_bound_mult+1)*radius[0],
                        (unet_bound_mult-1)*radius[1]:(unet_bound_mult+1)*radius[1], :].astype(float)
        print(f"Seg time: {time.time() - tic}")
        tic = time.time()

        compute_vectors = precompute_membrane_vectors(mem_to_run, mem_to_run, precomp_file_path, 3)
        print(f"Convolution time: {time.time() - tic}")

        sensor_list = [
            (agents.sensor.BrownianMotionSensor(), .1),
            (agents.sensor.PrecomputedSensor(), .2),
            (agents.sensor.MembraneSensor(mem_to_run), 0),
            ]

        pos_histories, seg_ids, agent_ids=run_agents(mem=mem_to_run, sensor_list=sensor_list,
                                                    precompute_fn=precomp_file_path+'.npz',
                                                    max_vel=.9,n_steps=500,segmentation=seg, root_id=root_id)
        # find merges of agents from root_id
        pos_matrix = create_post_matrix(pos_histories, mem_to_run.shape)
        merges = merge_paths(pos_histories,seg_ids, ep, root_id)
        # if filter_merge:
        #     for key in merges:
        if merges_root.shape[0] > 0:
            merges_root = merges[(merges.M1 == root_id) | (merges.M2 == root_id)]
            if merges_root.iloc[0].M1 == root_id:
                merges_root.rename({"M1":"root_id", "M2":"segment_id"})
            else:
                merges_root.rename({"M2":"root_id", "M1":"segment_id"})
            client = CAVEclient('minnie65_phase3_v1')
            merges_root.to_csv(f"./merges_root_{root_id}_{endpoint}.csv")
            for seg_id in merges_root.unique():
                # Double check
                if get_soma(seg_id, client) > 0:
                    print("FOUND SOMA in DOUBLE CHECK")
                    merges_root = merges_root[merges_root.segment_id != seg_id]
            weights_dict = dict(zip(merges_root.seg_id, merges_root.Weight))
        else:
            weights_dict = {}
        if save == "pd":
            merges.to_csv(f"./merges_{root_id}_{endpoint}.csv")
            merges_root.to_csv(f"./merges_root_{root_id}_{endpoint}.csv")
        if save == "nvq":
            import neuvueclient as Client
            print(time.time()-tic)
            agents_dict = {"Root_id": root_id, "Endpoint":endpoint,
                           "Nucleus_id":nucleus_id, "time":time_point, "Merges": weights_dict}

            C = Client.NeuvueQueue("http://neuvuequeue-server/")
            C.post_agents(**agents_dict)

        if save == "sqs":

            weights_dict = dict(zip(merges_root.seg_id, merges_root.Weight))

            agents_dict = {"Root_id": root_id, "Endpoint":endpoint, "Hash":hash, "Merges": weights_dict}
            att_dict = {
                            'root_id': {
                                'StringValue': str(root_id),
                                'DataType': 'String',
                            },
                            'endpoint': {
                                'StringValue': ','.join(endpoint.astype(str)),
                                'DataType': 'String',
                            },
                            'merges': {
                                'StringValue': str(merges_root.seg_id),
                                'DataType': 'String',
                            },
                            'weights': {
                                'StringValue': str(merges_root.Weight),
                                'DataType': 'String',
                            }
                        }
            sqs.send_one("Agents", att_dict)

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

if __name__ == "__main__":
    n = sys.argv[1]
    drive(n, **dict(arg.split('=') for arg in sys.argv[3:]))