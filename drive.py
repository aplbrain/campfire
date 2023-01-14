from logging import root
import pickle
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
    high_confidence_tips, med_confidence_tips, low_confidence_tips  = endpoints_from_rid(root_id)
    if delete:
        root_id_msg.delete()
    if high_confidence_tips is None:
        return
    if save == 'sqs':
        queue_url_endpoints = sqs.get_or_create_queue("Endpoints")

        entries=sqs.construct_endpoint_entries(high_confidence_tips, root_id)
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
        if high_confidence_tips.shape[0] == 0 and med_confidence_tips.shape[0] == 0 and low_confidence_tips.shape[0]:
            s1 = nvc_post_point(C, np.array([0, 0, 0]), "Justin", namespace, "no_tips_found", time_point, metadata)
        if high_confidence_tips.shape[0] > 0:
            s1 = nvc_post_point(C, (high_confidence_tips).astype(int), "Justin", namespace, "error_high_confidence_", time_point, metadata)
        else:
            s1 = -1
        if med_confidence_tips.shape[0] > 0:
            s2 = nvc_post_point(C, (med_confidence_tips).astype(int), "Justin", namespace, "error_med_confidence", time_point, metadata)
        else:
            s2 = -1
        if low_confidence_tips.shape[0] > 0:
            s3 = nvc_post_point(C, (low_confidence_tips).astype(int), "Justin", namespace, "error_low_confidence", time_point, metadata)
        else:
            s3 = -1
        print("Posted", s1, s2, s3)
    return high_confidence_tips

def errors_defects_facets(queue_url_rid, namespace='Errors_defects', save='nvq', delete=False):
    from tip_finding.tip_finding import error_locs_defects
    import pickle

    root_id_msg = sqs.get_job_from_queue(queue_url_rid)
    soma_id, root_id = np.fromstring(root_id_msg.body, dtype=np.uint64, sep=':')
    #soma_id = root_id_soma[:root_id_soma.find(':')]
    #root_id = root_id_soma[root_id_soma.find(':') + 1:]

    print("RID Processing", root_id)
    st = pickle.load(open('soma_table.p', 'rb'))

    r = error_locs_defects(root_id, soma_id, soma_table=st)
    if r is None:
        return
    sorted_encapsulated_send, facets_send_final, errors_send, errors_tips_send, rf, re, re_ep = r


    # if save == 'sqs':
    #     queue_url_endpoints = sqs.get_or_create_queue("Endpoints")

    #     entries=sqs.construct_endpoint_entries(high_confidence_tips, root_id)
    #     while(len(entries) > 0):
    #         entries_send = entries[:10]
    #         entries = entries[10:]
    #         sqs.send_batch(queue_url_endpoints, entries_send)
    if save == 'nvq':
        import neuvueclient as Client
        time_point = datetime.datetime.now()

        metadata = {'time':str(time_point), 
                    'root_id':str(root_id),
                    }
        C = Client.NeuvueQueue("https://queue.neuvue.io")
        time_point = time.time()#str(datetime.datetime.now())
        if sorted_encapsulated_send.shape[0] == 0 and facets_send_final.shape[0] == 0 and errors_send.shape[0]:
            s1 = nvc_post_point(C, np.array([0, 0, 0]), "Justin", namespace, "no_tips_found", time_point, metadata, [])
        if sorted_encapsulated_send.shape[0] > 0:
            s1 = nvc_post_point(C, (sorted_encapsulated_send).astype(int), "Justin", namespace, "encapsulated_points", time_point, metadata, [])
        else:
            s1 = -1
        if facets_send_final.shape[0] > 0:
            s2 = nvc_post_point(C, (facets_send_final).astype(int), "Justin", namespace, "flats", time_point, metadata, rf)
        else:
            s2 = -1
        if errors_send.shape[0] > 0:
            s3 = nvc_post_point(C, (errors_send).astype(int), "Justin", namespace, "general_error_locs", time_point, metadata, re)
        else:
            s3 = -1
        if errors_tips_send.shape[0] > 0:
            s4 = nvc_post_point(C, (errors_tips_send).astype(int), "Justin", namespace, "tip_error_locs", time_point, metadata, re_ep)
        else:
            s4 = -1
        #print("Posted", s1, s2, s3)
    if delete:
        root_id_msg.delete()
    return sorted_encapsulated_send

def run_endpoints(end, namespace="tips", save='nvq', delete=False):

    queue_url_rid = sqs.get_or_create_queue("Root_ids_functional_prod")
    print('Run endpoints', end, queue_url_rid)
    #root_ids = pickle.load(open('root_ids.p', 'rb'))
    n_root_id = 0
    while n_root_id < end or end == -1:
        print("N", n_root_id)
        errors_defects_facets(queue_url_rid, namespace, save, delete)
        n_root_id+=1
        print("Done", n_root_id)
    return 1

def nvc_post_point(C, points, author, namespace, name, status, metadata, weights):
    import copy
    base_metadata = copy.deepcopy(metadata)
    if len(points.shape) == 1:
        if len(weights) > 0:
            metadata['weight'] = weights
        C.post_point([str(p) for p in points], author, namespace, name, status, metadata)
    elif len(points.shape) == 0:
        return 0
    else:
        for i, pt in enumerate(points):
            metadata = copy.deepcopy(base_metadata)
            if len(weights) > 0:
                print(weights.shape, points.shape)
                metadata['weight'] = weights[i]
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
        return ext, success
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
    n_machines = 2
    # print("points", points)s
    while True:
        for p in range(points.shape[0]):
            row = points.iloc[p]
            rid = int(row.metadata['root_id'])
        
            if row.created.second % n_machines == 0:
                continue

            namespace_save = f"{namespace_agt}_{row.type[-1]}"
            print(p, points.iloc[p].coordinate, rid, namespace_save)
            pt = np.array(row.coordinate).astype(int)
            ext, s = segment_points(rid, pt, idx[p], radius=radius, resolution=rez, unet_bound_mult=unet_bound_mult, save=save, device=device, namespace=namespace_save, direction_test=direction_test)
            if s < 0:
                ext.save_agent_merges(s)
                print("saved_error")
                continue
            ext.save_agent_merges()
            print("Done", rid, p, ext.merges)
        points = get_points_nvc({"namespace":namespace, 'type':['error_high_confidence_thick', 'error_high_confidence_thin'], 'agents_status':'open'})
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

def error_fill_loop(namespace):
    from caveclient import CAVEclient
    C = CAVEclient.NeuvueQueue("https://queue.neuvue.io")
    points = get_points_nvc({"namespace":namespace, 'type':['encapsulated_points'], 'agents_status':'open'})
    idx = points.index

    for i, point in enumerate(points):
        rid = int(point.metadata['root_id'])
        center = np.array(point.coordinate) / [2,2,1]
        segs = error_fill(center, rid)
        md = point.metadata
        md['seg_merges'] = segs
        C.patch_point(idx[i], agents_status='extension_completed', metadata = md)


def error_fill(center, root_id):
    import fill_voids
    from agents import data_loader
    from caveclient import CAVEclient

    client = CAVEclient('minnie65_phase3_v1')
    radius = (200,200,15)

    seg = np.squeeze(data_loader.get_seg(center[0] - radius[0],
                center[0] + radius[0],
                center[1] - radius[1],
                center[1] + radius[1],
                center[2] - radius[2],
                center[2] + radius[2]))
    u, ns = np.unique(seg, return_counts=True)

    u = u[np.argsort(ns)][::-1]
        
    seg_ids = np.array(u).astype(np.int64)
    root_id_newest = client.chunkedgraph.get_latest_roots(root_id)

    if not root_id_newest in seg_ids:
        print('int')
        out_of_date_mask = client.chunkedgraph.is_latest_roots(seg_ids)
        seg_ids = np.array(seg_ids)
        out_of_date_seg_ids = seg_ids[~(out_of_date_mask)]
        seg_dict={}

        for s in out_of_date_seg_ids:
            print(s)
            if s == 0:
                seg_dict[0] = 0
                continue
                
            new_s = client.chunkedgraph.get_latest_roots(s)
            
            if root_id in new_s:
                seg_mask = s
                break
                
            seg_ids[seg_ids == s] = new_s[-1]
            seg_dict[new_s[-1]]=s
        for s in seg_ids[out_of_date_mask]:
            seg_dict[s] = s
    mask = seg == seg_mask
    filled = fill_voids.fill(mask)
    diff = ~mask * filled
    to_return = list(np.unique(seg[diff]))
    return to_return

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
        error_fill_loop('Tip_detect_defects_v3')
