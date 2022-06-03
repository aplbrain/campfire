from caveclient import CAVEclient
from intern import array
import pickle
import numpy as np
from agents import data_loader
from cloudvolume import CloudVolume
from membrane_detection import membranes
from agents.scripts import precompute_membrane_vectors, create_post_matrix, merge_paths, get_soma # get_syn_counts
import agents.sensor
from agents.run import run_agents
import aws.sqs as sqs
import sys
import time
import ast

def drive(n, radius=(200,200,20), resolution=(8,8,40), unet_bound_mult=2, ep='sqs', save='sqs',device='cpu',
          cnn_weights='thick',filter_merge=True,delete=True, endp=(0,0,0), nucleus_id=0, root_id=0, time_point=0):
    # queue_url_endpts = sqs.get_or_create_queue('Endpoints_Test')
    vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
    resolution = [int(x) for x in resolution]
    radius = [int(x) for x in radius]
    unet_bound_mult = float(unet_bound_mult)
    unet_bound = [int(x*float(unet_bound_mult)) for x in radius]
    n = int(n)
    ep_param = ep
    if n == -1:
        n = int(1e10)
    for _ in range(n):
        print(_)
        tic1=time.time()
        if ep_param == 'sqs':
            ep_msg = sqs.get_job_from_queue(queue_url_endpts)
            root_id = int(ep_msg.message_attributes['root_id']['StringValue'])
            nucleus_id = int(ep_msg.message_attributes['nucleus_id']['StringValue'])
            time_point = int(ep_msg.message_attributes['time']['StringValue'])
            ep = [float(p) for p in ep_msg.body.split(',')]
            print(root_id, nucleus_id, ep)
            if delete:
                ep_msg.delete()
        else:
            root_id = root_id
            nucleus_id = nucleus_id
            time_point = time_point
            ep = [float(p) for p in endp] 
        endpoint = np.divide(ep, resolution).astype('int')
        precomp_file_path = f"./precompute_{root_id}_{endpoint}"
        bound  = (endpoint[0] - radius[0], 
                    endpoint[0] + radius[0],
                    endpoint[1] - radius[1],
                    endpoint[1] + radius[1],
                    endpoint[2] - radius[2],
                    endpoint[2] + radius[2])
        
        bound_EM  = (endpoint[0] - unet_bound[0], 
                    endpoint[0] + unet_bound[0],
                    endpoint[1] - unet_bound[1],
                    endpoint[1] + unet_bound[1],
                    endpoint[2] - radius[2],
                    endpoint[2] + radius[2])
        seg = np.squeeze(data_loader.get_seg(*bound))
        tic = time.time()
        
        vol = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
        try:
            em = np.squeeze(vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3], bound_EM[4]:bound_EM[5]])
        except OutOfBoundsError:
            continue
        mem_seg = membranes.segment_membranes(em, pth="./membrane_detection/best_metric_model_segmentation2d_dict.pth", device_s=device)
        mem_to_run = mem_seg[int((unet_bound_mult-1)*radius[0]):int((unet_bound_mult+1)*radius[0]),
                        int((unet_bound_mult-1)*radius[1]):int((unet_bound_mult+1)*radius[1]), :].astype(float)
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
        pos_matrix, merge_d = create_post_matrix(pos_histories, seg, mem_to_run.shape)
        merges = merge_paths(pos_histories,seg_ids, ep, root_id)
        # if filter_merge:
        #     for key in merges:
        if merges.shape[0] > 0:
            client = CAVEclient('minnie65_phase3_v1')
            # merges.to_csv(f"./merges_root_{root_id}_{endpoint}.csv")
            # for seg_id in merges.seg_id.unique():
            #     # Double check
            #     if get_soma(seg_id, client) > 0:
            #         print("FOUND SOMA in DOUBLE CHECK")
            #         merges = merges[merges.seg_id != seg_id]
            seg_ids = [int (x) for x in merges.seg_id]
            
            weights = [int(x) for x in merges.Weight]
            weights_dict = dict(zip(seg_ids, weights))
        else:
            weights_dict = {}
        if save == "pd":
            merges.to_csv(f"./merges_{root_id}_{endpoint}.csv")
        if save == "nvq":
            import neuvueclient as Client
            duration = time.time()-tic1
            print("DURATION", duration, root_id, endpoint, "\n")
            metadata = {'time':str(time_point), 'duration':str(duration), 'device':device,'bbox':[str(x) for x in bound], 'bbox_em':[str(x) for x in bound_EM]}
            C = Client.NeuvueQueue("https://queue.neuvue.io")
            end = [str(x) for x in endpoint]
            print(str(root_id), str(nucleus_id), end, weights_dict, metadata)
            C.post_agent(str(root_id), str(nucleus_id), end, weights_dict, metadata)
            
            vol = array("bossdb://microns/minnie65_8x8x40/membranes", axis_order="XYZ")
            pickle.dump(mem_seg, open(f"./data/INTERNmem_{duration}_{root_id}_{endpoint}.p", "wb"))

            pickle.dump(bound_EM, open(f"./data/INTERNBOUNDS_{duration}_{root_id}_{endpoint}.p", "wb"))
            vol[bound_EM[0]:bound_EM[1], bound_EM[2]:bound_EM[3],bound_EM[4]:bound_EM[5]] = mem_seg.astype(np.uint64)
        
    return mem_seg, merges, pos_matrix, seg_ids, em, seg, compute_vectors


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
    drive(n, **dict(arg.split('=') for arg in sys.argv[2:]))
