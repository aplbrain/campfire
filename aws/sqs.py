""" 
Utility functions for sending and receiving messages from AWS SQS. 
"""

import boto3
import time
from typing import Iterable, List, Dict
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s ', 
    datefmt='%m/%d/%Y %I:%M:%S %p', 
    level=logging.INFO
    )

def send_batch(queue_url, message_batch, **kwargs):
    """
    Function to send messages and ensure all messages sent successfully

    Args:
        sqs (botocore.client) : sqs client
        queue_url (str) : queue url 
        message_batch (list) : list of messages. each message is a dict with an id and body keys.
    
    Returns: 
        None
    """
    session = boto3.Session(region_name="us-east-1", **kwargs)
    sqs = session.client("sqs")

    retries = 3
    while retries > 0: 
        response = sqs.send_message_batch(QueueUrl=queue_url, Entries=message_batch)
        if len(response.get('Failed', [])) == 0:
            break 
        retries -= 1
        failed_ids = [i['Id'] for i in response['Failed']]
        message_batch = [i for i in message_batch if i['Id'] in failed_ids]
    
    if len(response.get('Failed', [])) > 0:
        raise Exception(response['Failed'])

def construct_endpoint_entries(
    endpoint_batch: Iterable[List], 
    root_id: str,
    t: str,
    nucleus_id: str
    ) -> Dict:
    """
    Create a message batch entry for the branch tip endpoints for one root_id. 

    Args:
        endpoint_batch (List[str]): list of xyz endpoints 
    
    Returns:
        entries (List[Dict]): list of message batch entries
    """
    entries = []
    for i, endpoint in enumerate(endpoint_batch):
        message_body = ','.join(endpoint.astype(str))
        entries.append({
            'Id': str(i),
            'MessageBody': message_body,
            'MessageAttributes': {
                'time': {
                    'StringValue': str(t),
                    'DataType': 'String',
                },
                'nucleus_id': {
                    'StringValue': str(nucleus_id),
                    'DataType': 'String',
                },
                'root_id': {
                    'StringValue': str(root_id),
                    'DataType': 'String',
                }
            }
        })
    return entries

def construct_rootid_entries(
    root_id_batch: Iterable[List], 
    ) -> Dict:
    """
    Create a message batch entry for the root_ids. 

    Args:
        root_id_batch (List[str]): list of root ids
        cloudvolume_path: cloudvolume path to upload to
    
    Returns:
        entries (List[Dict]): list of message batch entries
    """
    entries = []
    for i, root_id in enumerate(root_id_batch):
        message_body = str(root_id)
        entries.append({
            'Id': str(i),
            'MessageBody': message_body,
        })
    return entries

def get_job_from_queue(queue_url, **kwargs):
    """
    Gets a job from the queue.
    """
    session = boto3.Session(region_name="us-east-1", **kwargs)
    sqs = session.resource("sqs")
    
    # get queue from url:
    queue = sqs.Queue(queue_url)

    # Get a single message from the queue:
    response = queue.receive_messages(
        MaxNumberOfMessages=1, WaitTimeSeconds=1, MessageAttributeNames=["All"]
    )

    if len(response) == 0:
        return None
    message = response[0]
    return message


def get_or_create_queue(queue_name:str, **kwargs):
    """
    Creates a queue with the given name. Returns the queue url.
    """
    session = boto3.Session(region_name="us-east-1", **kwargs)
    # Create the queue:
    sqs = session.client("sqs")
    
    # see if the queue already exists:
    queues = sqs.list_queues(QueueNamePrefix=queue_name)
    if "QueueUrls" in queues:
        return queues["QueueUrls"][0]
    else:
        print("Creating a new queue...")
        return sqs.create_queue(QueueName=queue_name)["QueueUrl"]