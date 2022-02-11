"""
Script to set parameters and run an agent run
"""

import time
import numpy as np
from swarm import Swarm
from scripts import precompute_membrane_vectors, load_membrane_vectors, create_queue


def run_agents(**kwargs):
    """
    Primary driver function for agents- organizes variables then steps the swarm n_steps times
    Arguments:
        **kwargs: all of the parameters listed below TODO should set default params
    Returns:
        None: all relavent output is pickled to file.
    """
    tic = time.time()
    # Instantiate graph variables- tracking which agents collide and which synapses
    # Found by which agents
    dataset = "fib"
    # -------------------
    # Can iterate over kwargs.items(), (k) = v
    # Can also use namespace object with dict as input, in __init__ use __setatr__(k,v)
    mem = kwargs["mem"]
    sensor_list = kwargs["sensor_list"]
    precompute_fn = kwargs["precompute_fn"]
    max_vel = kwargs["max_vel"]
    n_steps = kwargs["n_steps"]
    seg = kwargs["segmentation"]
    # -------------------
    # Preparing Data, Starting Locs and Swarm

    data = load_membrane_vectors(precompute_fn)

    agent_queue = create_queue(
        data.shape, 0, sampling_type="synapse", segmentation=seg
    )

    print(f"Data Prep Time: {time.time() - tic}")

    tic = time.time()

    # Uncomment this line to add agent spawning linearly throughout the volume
    # agent_queue += create_queue(data.shape, n_pts_per_dim, sampling_type="lin")

    num_agents = len(agent_queue)
    # Spawn agents
    count = 1
    swarm = Swarm(
        data=data,
        mem=mem,
        agent_queue=agent_queue,
        sensor_list=sensor_list,
        num_agents=num_agents,
        max_velocity=max_vel,
        parallel=False,
    )
    count = swarm.get_count()
    print("\nAgent Spawning Prep Time", time.time() - tic)
    print("Number of Agents = ", count - 1)
    print("Agent-Steps to be taken:", (count - 1) * n_steps, end=" ")
    print("Projected Time:", (count - 1) * n_steps * 0.000080796 / 60, " minutes")
    # Run agents
    tic = time.time()
    for _ in range(n_steps):
        swarm.step()

    steps_time = time.time() - tic

    print("Steps Time", steps_time / 60, "mins")
    print(
        "Difference between projected and real:",
        (steps_time - (count - 1) * n_steps * 0.000080796) / 60,
        " minutes",
    )
    # Save out data to file
    pos_histories = [a.get_position_history() for a in swarm.agents]

