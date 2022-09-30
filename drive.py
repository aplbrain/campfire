from logging import root
import pandas as pd
import numpy as np
import time
import aws.sqs as sqs
import sys
import backoff
import datetime

#@backoff.on_exception(backoff.expo, Exception, max_tries=5)
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
        time_point = time.time()#str(datetime.datetime.now())
        if good_tips_thick.shape[0] > 0:
            s1 = nvc_post_point(C, (good_tips_thick).astype(int), "Justin", namespace, "error_high_confidence_thick", time_point, metadata)
        else:
            s1 = -1
        if good_tips_thin.shape[0] > 0:
            s2 = nvc_post_point(C, (good_tips_thin).astype(int), "Justin", namespace, "error_high_confidence_thin", time_point, metadata)
        else:
            s2 = -1
        if good_tips_bad_thick.shape[0] > 0:
            s3 = nvc_post_point(C, (good_tips_bad_thick).astype(int), "Justin", namespace, "error_low_confidence_thick", time_point, metadata)
        else:
            s3 = -1
        if good_tips_bad_thin.shape[0] > 0:
            s4 = nvc_post_point(C, (good_tips_bad_thin).astype(int), "Justin", namespace, "error_low_confidence_thin", time_point, metadata)
        else:
            s4 = -1
        if all_tips.shape[0] > 0:
            s5 = nvc_post_point(C, (all_tips).astype(int), "Justin", namespace, "all_tips", time_point, metadata)
        else:
            s5 = -1
        all_flat = np.array(all_flat)
        if all_flat.shape[0] > 0:
            s6 = nvc_post_point(C, (all_flat).astype(int), "Justin", namespace, "all_flats", time_point, metadata)
        else:
            s6 = -1
        print(s1, s2, s3, s4, s5, s6)
    return good_tips_thick 

def run_endpoints(end, namespace="tips", save='nvq', delete=False):
    queue_url_rid = sqs.get_or_create_queue("Root_ids_apical")
    n_root_id = 0
    while n_root_id < end or end == -1:
        endpoints(queue_url_rid, namespace, save, delete)
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

def segment_points(root_id, endpoint, point_id, radius=(200,200,30), resolution=(8,8,40), unet_bound_mult=1.5, save='pd',device='cpu',
                   nucleus_id=0, time_point=0, namespace='Agents', direction_test=True):
    
    if time_point == 0:
        time_point = datetime.datetime.now()

    tic = time.time()
    from extension import Extension as Ext
    from extension import save_merges
    ext = Ext(root_id, resolution, radius, unet_bound_mult, 
    device, save, nucleus_id, time_point, namespace, direction_test, point_id)
    toc = time.time()
    print("Ext Time", toc-tic)
    ext.get_bounds(endpoint)
    tic = time.time()
    print("Bounds Time", tic-toc)
    success = ext.gen_membranes(intern_pull=False, restitch_gaps=True)
    toc = time.time()
    print("Membranes Time", toc-tic)

    if success < 0:
        print("Errored out")
        ext.merges = pd.DataFrame({"Error": "Error"}, index=[0])
        ext.weights_dict = {}
        ext.mem_seg = 0
        ext.n_errors = 61
        return ext, 0
    # ext.run_agents(nsteps=1000)
    ext.dist_transform_merge()
    ext.distance_merge_save()
    # duration = time.time()-ext.tic1

    #save_merges(ext.save, ext.merges, ext.root_id[0], ext.nucleus_id, ext.time_point, ext.endpoint,
            #ext.weights_dict, ext.bound, ext.bound_EM, ext.mem_seg, ext.device, duration, ext.n_errors, ext.namespace)
    return ext, 1

def run_nvc_agents(namespace, namespace_agt, radius=(300,300,30), rez=(8,8,40), unet_bound_mult=1, save='nvq', device='cpu', direction_test=True):
    print(namespace, namespace_agt, radius, rez, unet_bound_mult, save, device, direction_test)
    points = get_points_nvc({"namespace":namespace, 'type':['error_high_confidence_thick', 'error_high_confidence_thin'], 'agents_status':'open'})
    idx = points.index
    # print("points", points)s
    for p in range(points.shape[0]):
        row = points.iloc[p]
        rid = int(row.metadata['root_id'])

        namespace_save = f"{namespace_agt}_{row.type[-1]}"
        print(p, points.iloc[p].coordinate, rid, namespace_save)
        pt = np.array(row.coordinate).astype(int)
        ext, s = segment_points(rid, pt, idx.iloc[p], radius=radius, resolution=rez, unet_bound_mult=unet_bound_mult, save=save, device=device, namespace=namespace_save, direction_test=direction_test)
        if s == 0:
            ext.save_agent_merges(True)
        ext.save_agent_merges()
        print("Done", rid, p, ext.merges)
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
        namespace=str(sys.argv[5])
        run_endpoints(end, namespace, save, delete)
    if mode == 'agents':
        run_nvc_agents(save=sys.argv[2], device=sys.argv[3], namespace=sys.argv[5], namespace_agt=sys.argv[4], rez=np.array([8,8,40]))
