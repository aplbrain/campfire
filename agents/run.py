"""
Script to set parameters and run an agent run
"""

import time
import numpy as np
from agents.swarm import Swarm
from agents.scripts import load_membrane_vectors, create_queue
import copy


def run_agents(**kwargs):
    """
    Primary driver function for agents- organizes variables then steps the swarm n_steps times
    Arguments:
        **kwargs: all of the parameters listed below TODO should set default params
    Returns:
        None: all relavent output is pickled to file.
    """

    tic = time.time()
    # -------------------
    # Can iterate over kwargs.items(), (k) = v
    # Can also use namespace object with dict as input, in __init__ use __setatr__(k,v)
    mem = kwargs["mem"]
    sensor_list = kwargs["sensor_list"]
    precompute_fn = kwargs["precompute_fn"]
    max_vel = kwargs["max_vel"]
    n_steps = kwargs["n_steps"]
    seg = kwargs["segmentation"]
    root_id = kwargs["root_id"]
    endpoint_nm = kwargs["endpoint_nm"]
    # -------------------
    # Preparing Data, Starting Locs and Swarm

    data = load_membrane_vectors(precompute_fn)
    agent_queue, polarity = create_queue(
        data.shape,
        100,
        sampling_type="extension_only",
        root_id=root_id,
        segmentation=seg,
        endpoint_nm=endpoint_nm,
    )
    if polarity == "Axon":
        return [], [root_id], []
    # Uncomment this line to add agent spawning linearly throughout the volume
    # agent_queue += create_queue(data.shape, n_pts_per_dim, sampling_type="lin")

    num_agents = len(agent_queue)
    # Spawn agents
    count = 1
    swarm = Swarm(
        data=data,
        mem=mem,
        seg=seg,
        agent_queue=agent_queue,
        sensor_list=sensor_list,
        num_agents=num_agents,
        max_velocity=max_vel,
        parallel=False,
    )
    count = swarm.get_count()
    print(count, "agents spawned")
    print("\nAgent Spawning Prep Time", time.time() - tic)
    # Run agents
    tic = time.time()
    for _ in range(n_steps):
        swarm.step()

    steps_time = time.time() - tic

    print("Steps Time", steps_time)
    # Save out data to file
    pos_histories = [a.get_position_history() for a in swarm.agents]
    seg_ids = [a.seg_id for a in swarm.agents]
    agent_ids = [a.agent_id for a in swarm.agents]
    return pos_histories, seg_ids, agent_ids
