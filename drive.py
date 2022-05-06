from logging import root
from spine_finding import spine_finding
from caveclient import CAVEclient
from cloudvolume import CloudVolume
import aws.sqs as sqs
import sys

def run_endpoints(root_id,radius=(100,100,10), resolution=(8,8,40), unet_bound_mult=2, return_opacity_seg=False,save_format='pickle',delete=False):
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

if __name__ == "__main__":
    mode = sys.argv[1]
    root_id = sys.argv[2]
    if mode == 'endpoints':
        run_endpoints(root_id, **dict(arg.split('=') for arg in sys.argv[3:]))