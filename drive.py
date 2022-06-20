import pandas as pd
import numpy as np
from tip_finding.tip_finding import tip_finder_decimation
from extension import Extension as Ext

def segment_series(root_id, endpoints, radius=(200,200,20), resolution=(2,2,1), unet_bound_mult=1.5, save='pd',device='cpu',
                   nucleus_id=0, time_point=0, threshold=8):
    hit_ids = [root_id]
    current_root_id = root_id
    
    merges = pd.DataFrame()
    ext = Ext(root_id, resolution, radius, unet_bound_mult, 
              device, save, nucleus_id, time_point)

    while True:
        try:
            endpoint = endpoints.pop()
        except IndexError:
            merges.to_csv("./merges_series")
            return merges
        ext.gen_membranes(endpoint)
        ext.run_agents()
        initial_merges = ext.merges
        merges = pd.concat([merges, initial_merges])
        initial_merges = initial_merges[initial_merges.Weight > threshold]
        initial_merges = initial_merges[initial_merges.Synapse_filter == False]
        initial_merges = initial_merges[initial_merges.Polarity_filter == False]
        initial_merges = initial_merges[np.logical_not(initial_merges.Extension.isin(hit_ids))]

        if initial_merges.shape[0] == 0:
            return
        highest_val = initial_merges.sort_values('Weight', ascending=False).iloc[0]
        current_root_id = highest_val.Extension
        hit_ids.append(current_root_id)
        next_endpoints = tip_finder_decimation(hit_ids)
        for ep in next_endpoints:
            endpoints.append(ep)

def segment_gt_points(radius=(200,200,20), resolution=(2,2,1), unet_bound_mult=1.5, save='pd',device='cpu',
                   nucleus_id=0, time_point=0, threshold=8):
    
    gt_df = pd.read_csv("./matched_gt.csv")

    merges = pd.DataFrame()
    for rid in gt_df.Root_id.unique():
            root_id = int(gt_df.iloc[0].Root_id)
            ext = Ext(root_id, resolution, radius, unet_bound_mult, 
              device, save, nucleus_id, time_point)
            filt_df = gt_df[gt_df.Root_id == rid]
            for i in range(filt_df.shape[0]):
                row = filt_df.iloc[i]

                endpoint = eval(row.EP)

                ext.gen_membranes(endpoint)
                ext.run_agents()
                initial_merges = ext.merges
                initial_merges.to_csv(f"merges_{root_id}_{row.EP}.csv")
                merges = pd.concat([merges, initial_merges])
                ext.save_agent_merges()
                # initial_merges = initial_merges[initial_merges.Weight > threshold]
                # initial_merges = initial_merges[initial_merges.Synapse_filter == False]
                # initial_merges = initial_merges[initial_merges.Polarity_filter == False]
                # initial_merges = initial_merges[np.logical_not(initial_merges.Extension.isin(hit_ids))]
                # if initial_merges.shape[0] == 0:
                #     return
                # highest_val = initial_merges.sort_values('Weight', ascending=False).iloc[0].Extension
                # initial_merges = initial_merges.sort_values("Angle", ascending=False).iloc[0]
                # highest_val_traj = initial_merges.Extension
                # current_root_id = highest_val.Extension
                # next_endpoints = tip_finder_decimation(hit_ids)
    merges.to_csv("full_merges")

def sqs_agents(radius=(200,200,20), delete=False):
    root_id, nucleus_id, time_point, endp = utils.get_endpoint('sqs', delete, endp, root_id, nucleus_id, time_point)
    merges = drive()