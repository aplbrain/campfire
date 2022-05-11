from logging import root
from spine_finding import spine_finding
from caveclient import CAVEclient
from cloudvolume import CloudVolume
import aws.sqs as sqs
import sys

def run_endpoints(root_id, save_format='pickle',delete=False,return_skel=False):
    # SPINE FINDING

    client = CAVEclient('minnie65_phase3_v1')
    if root_id == 'sqs':
        queue_url = sqs.get_or_create_queue("Root_ids_endpoints")
        root_id_msg = sqs.get_job_from_queue(queue_url)
        root_id = int(root_id_msg.body)
    print("ROOT_ID", root_id)
    end_points = spine_finding.find_endpoints(root_id,
                               client=client,
                               refine='all',
                               root_point_resolution=[4, 4, 40],
                               collapse_soma=True,
                               n_parallel=8)

    if save_format == 'sqs':
        queue_url = sqs.get_or_create_queue('Endpoints')
        entries=sqs.construct_endpoint_entries(end_points, root_id)
        while(len(entries) > 0):
            entries_send = entries[:10]
            entries = entries[10:]
            sqs.send_batch(queue_url, entries_send)
        if root_id == 'sqs' and delete:
            root_id_msg.delete()
    elif save_format == 'pickle':
        import pickle
        pickle.dump(end_points, open(f"./data/{root_id}_endpoints.p", "wb"))

def run_endpoints_sqs(root_id, end, save_format='pickle',delete=False,return_skel=False):
    client = CAVEclient('minnie65_phase3_v1')
    queue_url_rid = sqs.get_or_create_queue("Root_ids_endpoints")
    queue_url_endpts = sqs.get_or_create_queue('Endpoints')

    root_id_msg = sqs.get_job_from_queue(queue_url_rid)
    root_id = int(root_id_msg.body)
    n_root_id = 0
    if end == -1:
        end = 1e10
    while n_root_id < end:
        print("ROOT_ID", root_id)
        end_points = spine_finding.find_endpoints(root_id,
                                client=client,
                                refine='all',
                                root_point_resolution=[4, 4, 40],
                                collapse_soma=True,
                                n_parallel=8)

        entries=sqs.construct_endpoint_entries(end_points, root_id)
        while(len(entries) > 0):
            entries_send = entries[:10]
            entries = entries[10:]
            sqs.send_batch(queue_url_endpts, entries_send)
        if delete:
            root_id_msg.delete()

if __name__ == "__main__":
    mode = sys.argv[1]
    root_id = sys.argv[2]
    if mode == 'endpoints':
        run_endpoints(root_id, **dict(arg.split('=') for arg in sys.argv[3:]))