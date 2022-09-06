from curses import meta
import pandas as pd
import numpy as np
from tip_finding.tip_finding import endpoints_from_rid
from extension import Extension as Ext
import time
import aws.sqs as sqs
import sys

def endpoints(queue_url_rid, save='nvq', delete=False):
    root_id_msg = sqs.get_job_from_queue(queue_url_rid)
    root_id = np.fromstring(root_id_msg.body, dtype=np.uint64, sep=',')[0]
    tips_thick, tips_thin, thru_branch_tips, tip_no_flat_thick, tip_no_flat_thin, flat_no_tip  = endpoints_from_rid(root_id)
    if delete:
        root_id_msg.delete()
    if save == 'sqs':
        queue_url_endpoints = sqs.get_or_create_queue("Endpoints")

        entries=sqs.construct_endpoint_entries(tips_thick, root_id)
        while(len(entries) > 0):
            entries_send = entries[:10]
            entries = entries[10:]
            sqs.send_batch(queue_url_endpoints, entries_send)
    elif save == 'nvq':
        import datetime
        import neuvueclient as Client
        time_point = datetime.datetime.now()

        metadata = {'time':str(time_point), 
                    'root_id':root_id,
                    }
        C = Client.NeuvueQueue("https://queue.neuvue.io")
        for pt in tips_thick:
            C.post_point(list(pt), 'Justin', 'Tip_detection', 'error_tip_thick', 0, metadata, True)
        for pt in tips_thin:
            C.post_point(list(pt), 'Justin', 'Tip_detection', 'error_tip_thin', 0, metadata, True)
        for pt in thru_branch_tips:
            C.post_point(list(pt), 'Justin', 'Tip_detection', 'error_tip_branch', 0, metadata, True)
        for pt in tip_no_flat_thick:
            C.post_point(list(pt), 'Justin', 'Tip_detection', 'tip_thick', 0, metadata, True)
        for pt in tip_no_flat_thin:
            C.post_point(list(pt), 'Justin', 'Tip_detection', 'tip_thin', 0, metadata, True)
        for pt in flat_no_tip:
            C.post_point(list(pt), 'Justin', 'Tip_detection', 'errorloc', 0, metadata, True)
    return tips_thick 

def run_endpoints(end, save='nvq', delete=False):
    print(end, save, delete, "SDg")
    sdfs
    queue_url_rid = sqs.get_or_create_queue("Root_ids_endpoints")
    n_root_id = 0
    while n_root_id < end or end == -1:
        endpoints(queue_url_rid, save, delete)
        n_root_id+=1
    return 1

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
        success = ext.gen_membranes(endpoint)
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

def segment_gt_points(radius=(200,200,30), resolution=(2,2,1), unet_bound_mult=1.5, save='pd',device='cpu',
                   nucleus_id=0, time_point=0, threshold=8):
    ct=0
    gt_df = pd.read_csv("./matching_master.csv")
    merges = pd.DataFrame()
    print("Unique Ids",gt_df.R.unique())
    for root_id in gt_df.R.unique():
        if root_id == 864691136577830164 or root_id == 864691135683615218:
            print("Skipping",root_id)
            continue
        filt_df = gt_df[gt_df.R == root_id]
        print("Shape", filt_df.shape[0])
        for i in range(filt_df.shape[0]):
            tic = time.time()
            row = filt_df.iloc[i]
            endpoint = eval(row.G1)
            print("Point", i, ct, root_id, endpoint)
            ext = Ext(root_id, resolution, radius, unet_bound_mult, 
            device, save, nucleus_id, time_point, endpoint)
            toc = time.time()
            print("Ext Time", toc-tic)
            ext.get_bounds(endpoint)
            tic = time.time()
            print("Bounds Time", tic-toc)
            success = ext.gen_membranes(restitch_gaps=True, restitch_linear=True)
            toc = time.time()
            print("Membranes Time", toc-tic)

            if success < 0:
                print("Errored out")
                ext.merges = pd.DataFrame({"Error": "Error"}, index=[0])
                ext.weights_dict = {}
                ext.mem_seg = 0
                ext.n_errors = 61
                ext.save_agent_merges()
                ct+=1
                continue
            ext.run_agents(nsteps=1000)
            initial_merges = ext.merges
            initial_merges.to_csv(f"merges_{root_id}_{row.G1}.csv")
            merges = pd.concat([merges, initial_merges])
            ext.save_agent_merges()
            ct+=1

def sqs_agents(radius=(200,200,20), delete=False):
    root_id, nucleus_id, time_point, endp = utils.get_endpoint('sqs', delete, endp, root_id, nucleus_id, time_point)
    merges = drive()

if __name__ == "__main__":
    # Wraps the command line arguments
    mode = sys.argv[1]

    if mode == 'tips':
        end = int(sys.argv[2])
        save = sys.argv[3]
        delete = bool(sys.argv[4])
        run_endpoints(end, save, delete)
    if mode == 'agents':
        segment_gt_points()