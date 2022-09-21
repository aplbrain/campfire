from logging import root
import pandas as pd
import numpy as np
import time
import aws.sqs as sqs
import sys
import backoff
import datetime

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def endpoints(queue_url_rid, namespace='Errors_GT', save='nvq', delete=False):
    from tip_finding.tip_finding import endpoints_from_rid

    root_id_msg = sqs.get_job_from_queue(queue_url_rid)
    root_id = np.fromstring(root_id_msg.body, dtype=np.uint64, sep=',')[0]
    print("RID", root_id)
    good_tips_thick, good_tips_thin, good_tips_bad_thick, good_tips_bad_thin, all_tips, all_flat  = endpoints_from_rid(root_id)
    if delete:
        root_id_msg.delete()
    if save == 'sqs':
        queue_url_endpoints = sqs.get_or_create_queue("Endpoints")

        entries=sqs.construct_endpoint_entries(good_tips_thick, root_id)
        while(len(entries) > 0):
            entries_send = entries[:10]
            entries = entries[10:]
            sqs.send_batch(queue_url_endpoints, entries_send)
    elif save == 'nvq':
        import neuvueclient as Client
        time_point = datetime.datetime.now()

        metadata = {'time':str(time_point), 
                    'root_id':str(root_id),
                    }
        C = Client.NeuvueQueue("https://queue.neuvue.io")
        nvc_post_point(C, good_tips_thick.astype(int), "Justin", namespace, "error_high_confidence_thick", 0, metadata)
        nvc_post_point(C, good_tips_thin.astype(int), "Justin", namespace, "error_high_confidence_thin", 0, metadata)
        nvc_post_point(C, good_tips_bad_thick.astype(int), "Justin", namespace, "error_low_confidence_thick", 0, metadata)
        nvc_post_point(C, good_tips_bad_thin.astype(int), "Justin", namespace, "error_low_confidence_thin", 0, metadata)
        nvc_post_point(C, all_tips.astype(int), "Justin", namespace, "all_tips", 0, metadata)
        nvc_post_point(C, all_flat.astype(int), "Justin", namespace, "all_flats", 0, metadata)
        
    return tips_thick 

def run_endpoints(end, save='nvq', delete=False):
    queue_url_rid = sqs.get_or_create_queue("Root_ids_endpoints")
    n_root_id = 0
    while n_root_id < end or end == -1:
        endpoints(queue_url_rid, save, delete)
        n_root_id+=1
    return 1

def nvc_post_point(C, points, author, namespace, name, status, metadata):
    if len(points.shape) == 1:
        C.post_point([str(p) for p in points], author, namespace, name, status, metadata)
    elif len(points.shape) == 0:
        return 0
    else:
        for pt in points:
            C.post_point([str(p) for p in pt], author, namespace, name, status, metadata)
    return 1

def get_points_nvc(filt_dict):
    import neuvueclient as Client
    C = Client.NeuvueQueue("https://queue.neuvue.io")
    return C.get_points(filt_dict)

def segment_points(root_id, endpoint, radius=(200,200,30), resolution=(8,8,40), unet_bound_mult=1.5, save='pd',device='cpu',
                   nucleus_id=0, time_point=0, namespace='Agents'):
    
    if time_point == 0:
        time_point = datetime.datetime.now()

    tic = time.time()
    from extension import Extension as Ext
    ext = Ext(root_id, resolution, radius, unet_bound_mult, 
    device, save, nucleus_id, time_point, endpoint, namespace)
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
        return ext, 0
    ext.run_agents(nsteps=1000)
    return ext, 1

def run_nvc_agents(namespace, namespace_agt, radius=(300,300,30), rez=(8,8,40), unet_bound_mult=1.5, save='nvq', device='cpu'):
    print(namespace, namespace_agt, radius, rez, unet_bound_mult, save, device)
    points = get_points_nvc({"namespace":namespace})
    # print("points", points)s
    for p in range(points.shape[0]):
        print(points.iloc[p].coordinate)
        row = points.iloc[p]
        rid = int(row.metadata['root_id'])
        pt = np.array(row.coordinate).astype(int)
        ext, s = segment_points(rid, pt, radius=radius, resolution=rez, unet_bound_mult=unet_bound_mult, save=save, device=device, namespace=namespace_agt)
        if s == 0:
            continue
        ext.save_agent_merges()
        print("Done". rid, p, ext.merges.iloc[0])
    return 1

def segment_gt_points(radius=(200,200,30), resolution=(2,2,1), unet_bound_mult=1.5, save='pd',device='cpu',
                   nucleus_id=0, time_point=0):
    from extension import Extension as Ext

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
            row = filt_df.iloc[i]
            endpoint = eval(row.G1)
            ext, s = segment_points(root_id, endpoint, radius, resolution)
            if s == 0:
                continue
            ext.merges.to_csv(f"merges_{root_id}_{row.G1}.csv")
            merges = pd.concat([merges, ext.merges])
            ct+=1


if __name__ == "__main__":
    # Wraps the command line arguments
    mode = sys.argv[1]
    print("args", sys.argv)
    if mode == 'tips':
        end = int(sys.argv[2])
        save = sys.argv[3]
        delete = bool(sys.argv[4])
        run_endpoints(end, save, delete)
    if mode == 'agents':
        run_nvc_agents(save=sys.argv[2], device=sys.argv[3], namespace=sys.argv[5], namespace_agt=sys.argv[4])
