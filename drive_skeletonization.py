from spine_finding import spine_finding
from caveclient import CAVEclient
import numpy as np
import aws.sqs as sqs
import sys

def run(end,delete=False):
    """
    Find euclidean-space skeleton end point vertices from the pychunkedgraph
    Parameters
    ----------
    end : uint64
        Number of iterations to run, where each iteration sends all root_id endpoints to SQS
    delete : bool
        If True, deletes the pulled root_id from SQS
    Returns
    -------
    None : Nonetype
        Endpoints are formatted and sent to SQS
    """
    
    # Client and SQS are instantiated. 
    # CAVEclient is pulling from public data v117
    # SQS Endpoints are specified AWS Simple Queue Service endpoints. 
    # These can be configured in the AWS account
    client = CAVEclient('minnie65_phase3_v1')
    queue_url_rid = sqs.get_or_create_queue("Root_ids_endpoints")
    queue_url_endpts = sqs.get_or_create_queue('Endpoints')

    n_root_id = 0
    # Send end = -1 to run forever
    while n_root_id < end or end == -1:

        root_id_msg = sqs.get_job_from_queue(queue_url_rid)
        # Casting root_id like this because casting to normal int loses precision
        root_id = np.fromstring(root_id_msg.body, dtype=np.uint64, sep=',')[0]
        print("ROOT_ID", root_id)
        # Extracting info from queue object. The queue holds strings, so they must be recast
        nucleus_id = int(root_id_msg.message_attributes['nucleus_id']['StringValue'])
        pt_position = int(root_id_msg.message_attributes['pt_position']['StringValue'])
        time = root_id_msg.message_attributes['time']['StringValue']
        # Current issue- when the algo crashes or is manually interrupted, we lose a root_id
        # This should be dealt with to only delete after the process is finished
        # The issue is Queue Timeout- if the queue doesn't receive a 'still working' signal,
        # it throws an error when you try to delete if enough time has elapsed
        if delete:
            root_id_msg.delete()
        # Call endpoint finder (this is a wrapper to pcg_skel)
        end_points = spine_finding.find_endpoints(root_id, nucleus_id, time, pt_position,
                                client=client,
                                refine='all',
                                root_point_resolution=[4, 4, 40],
                                collapse_soma=True,
                                n_parallel=8)

        entries=sqs.construct_endpoint_entries(end_points, root_id, time, nucleus_id, pt_position)
        # Send all entries in batches of 10 (the largest size that sqs will take)
        while(len(entries) > 0):
            entries_send = entries[:10]
            entries = entries[10:]
            sqs.send_batch(queue_url_endpts, entries_send)

        n_root_id+=1


if __name__ == "__main__":
    # Wraps the command line arguments
    run(int(sys.argv[1]),**dict(arg.split('=') for arg in sys.argv[3:]))
