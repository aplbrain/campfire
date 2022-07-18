#!/usr/bin/env python3
"""
Copyright 2017 The Johns Hopkins University Applied Physics Laboratory.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List
from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
from agents.agent import Agent
from agents.sensor import *
from tqdm import tqdm


class Swarm:
    """
    Base class for a swarm of agents.

    Includes logic for running swarm simulations.
    """

    def __init__(
        self,
        data: np.array = None,
        mem: np.array = None,
        seg: np.array = None,
        agent_queue: list = [],
        sensor_list: list = [],
        num_agents: int = 0,
        max_velocity: int = 1,
        parallel: bool = True,
        max_steps: int = None,
        isotropy: float = 1,
        show_tqdm: bool = False,
    ) -> None:
        """
        Create a new swarm.

        Arguments:
            data (np.array: None): Neuron Segmentation. Optional. Must be set
                manually before running the simulation.
            num_agents (int: 0): The number of agents to run. If unset, agents
                must be added manually with `Swarm#add_agent`.
            parallel (bool: True): Whether to try to run in parallel or not.
            max_steps (int): The maximum number of frames to execute.

        """
        # Set to `true` when data is present in the swarm
        self._data_exists = False
        self.data = None
        self.mem = mem
        # Denotes the resolution of the data relative to x/y.
        # Isotropy of 5 indicates that the z axis is 5 times more course than x/y.
        self.isotropy = isotropy
        # The number of agents in this swarm
        self.num_agents = num_agents

        # A boolean; true if the simulation should stop after a certain
        # number of steps
        self.uses_max_steps = max_steps is not None
        self.max_steps = max_steps

        # Should the simulation run in parallel?
        self._parallel = parallel

        # Populate data attribute if it's provided:
        if data is not None:
            self.set_data(data)

        # Adjacency matrix to track which seg_id's merge
        # 1st dimension tracks agents seeing each other
        # 2nd dimension tracks agents entering other segments
        # self.seg_map = np.zeros((*self.data.shape,2))
        # Agents publish their location here
        # self.agent_map = np.zeros(self.data.shape[:3])

        # self.agent_adjacency = {seg_id:i for(seg_id, i) in enumerate(np.unique(self.data))}
        self.id_map = {}
        self.agents = []
        self._count = 1
        # Populate the list of agents if empty Agents are requested
        # Don't need to check if I am spawning on a membrane
        # Think about about writing out positions to file using sqlite
        if show_tqdm:
            agent_range = tqdm(range(num_agents))
        else:
            agent_range = range(num_agents)

        for i in agent_range:
            init_pos = agent_queue[i]
            # print("POS", init_pos, self.data.shape)
            # print(self.data[init_pos].shape)
            seg_id = int(seg[init_pos[0], init_pos[1], init_pos[2]])
            # print("SEG", seg_id)
            self.add_agent(init_pos, seg_id, sensor_list, max_velocity)

    def set_data(self, data: np.array) -> None:
        """
        Set the data volume for the swarm to use.

        Arguments:
            data (np.array): The volume to segment

        """
        self.data = data
        self._data_exists = True

    def get_data(self) -> np.array:
        """
        Get the base data from the swarm.

        Arguments:
            None

        Returns:
            np.array: The data owned by the swarm

        """
        return self.data

    # Change to create agent and seperate add_agent that takes agent as input
    def add_agent(self, init_pos, seg_id, sensor_list, max_velocity) -> Agent:
        """
        Add a new agent.

        Arguments:
            agent (Agent): The agent to add

        Returns:
            int: The index of the newly added agent

        """
        agent = Agent(
            position=init_pos,
            agent_id=self._count,
            seg_id=seg_id,
            sensors=sensor_list,
            max_velocity=max_velocity,
        )
        self.id_map[self._count] = seg_id
        # change to list comprehension
        self._count += 1
        self.agents.append(agent)

        return agent

    def add_random(self, agent):
        """Add a random agent at a random non-membrane position
        Arguments:
            agent (Agent): The agent to add

        Returns:
            boolean: whether the operation succeeded or failed

        """
        shape = np.shape(self.mem)
        found_spot = 0
        while found_spot < 100:
            new_pos = [
                np.random.randint(0, shape[0]),
                np.random.randint(0, shape[1]),
                np.random.randint(0, shape[2]),
            ]
            if self.mem[new_pos[0], new_pos[1], new_pos[2]]:
                found_spot += 1
            else:
                self.add_agent(new_pos, agent.sensors, agent.max_velocity)
                self._count += 1
                return True
        return False

    def get_agents(self) -> List[Agent]:
        """
        Get a list of the agents in this swarm.

        Arguments:
            None

        Returns:
            List[Agent]: The agents associated with this swarm.

        """
        return self.agents

    def get_count(self) -> List[Agent]:
        """
        Get a count of the agents in this swarm.

        Arguments:
            None

        Returns:
            int: The number of agents associated with this swarm.

        """
        return self._count

    def step(self) -> bool:
        """
        Run a simulation step. Executes for each agent. Can be parallelized.

        Arguments:
            None

        Returns:
            bool: True if there are still any active agents; False if done.

        """
        if self.uses_max_steps:
            self.max_steps -= 1

            if self.max_steps == 0:
                return False

        any_alive = False
        # self.agent_map = np.zeros(self.data[:3])
        # Can run in parallel (the line below) or in series, using the `else`
        # block below.
        if self._parallel:
            # Define the step function call for each agent
            def _call_step(agent):
                return agent.step(self)

            # TODO: For some reason this is way slower right now JobLib(?)
            with ThreadPoolExecutor() as executor:
                # Map each agent to the result of its step (i.e if it survives)
                results = executor.map(_call_step, self.agents)
                return sum(results) > 0
        else:
            for agent in self.agents:
                # If the agent is alive, continue to run it
                if agent.alive:
                    any_alive = True
                    agent.step(self)

            # If no agents are alive, end the simulation
            if not any_alive:
                return False

        return True

    def export_run(self) -> List[List]:
        """
        Export a List of each agents' location history.

        Each history is a list of XYZ 3-tuples, so the schema as returned is:

            List[List[Number[3]]]

        Arguments:
            None

        Returns:
            List: A list of location histories. See function description for
                more information.

        """
        return [agent.get_position_history() for agent in self.get_agents()]
