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

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Dict
import numpy as np

class Sensor:
    """
    Abstract class for a sensor.

    Implement me!
    """

    @abstractmethod
    def get_vector(self, swarm_ptr, agent_ptr) -> None:
        """
        Get required data for this sensor. Has access to full swarm.

        Arguments:
            swarm_ptr (agents.Swarm): A pointer to the swarm to which this
                sensor's agent belongs
            agent_ptr (agents.Agent): A pointer to the agent

        Returns:
            numpy.array: An n-dimensional vector with direction and magnitude.

        """


# Each agent biased randomly
class BiasSensor(Sensor):
    """
    Sensor that generates constant bias towards one random direction to escape local minima
    """

    def __init__(self):

        self.vec = np.random.random((3,)) - 0.5

    def get_vector(self, swarm_ptr, agent_ptr) -> None:
        """
        Get required data for this sensor. Has access to full swarm.

        Arguments:
            swarm_ptr (agents.Swarm): A pointer to the swarm to which this
                sensor's agent belongs
            agent_ptr (agents.Agent): A pointer to the agent

        Returns:
            np.array: The new direction
        """
        return self.vec

class BrownianMotionSensor(Sensor):
    """
    A Brownian Motion simulator.

    Randomly generates a direction to introduce noise.
    """

    def get_vector(self, swarm_ptr, agent_ptr) -> None:
        """
        Get required data for this sensor. Has access to full swarm.

        Arguments:
            swarm_ptr (agents.Swarm): A pointer to the swarm to which this
                sensor's agent belongs
            agent_ptr (agents.Agent): A pointer to the agent

        Returns:
            np.array: The new direction (random)

        """
        vec = np.random.random((3,)) - 0.5
        return vec


class MembraneSensor(Sensor):
    """
    Kills agent when it hits membrane
    Can randomly spawn a new one afterwards in a random pos
    """

    def __init__(self, membrane) -> None:
        self.mems = membrane
        self.vec = np.zeros(3)

    def get_vector(self, swarm_ptr, agent_ptr):
        """
        Get required data for this sensor. Has access to full swarm.

        Arguments:
            swarm_ptr (agents.Swarm): A pointer to the swarm to which this
                sensor's agent belongs
            agent_ptr (agents.Agent): A pointer to the agent

        Returns:
            np.array: The new direction

        """
        # Somewhat redundant to check if I am spawning on a membrane
        pos = agent_ptr.get_position().astype(int)
        # try:
        # Commenting out try except for camp - most agents should be in the middle
        mem_or_not = self.mems[pos[0], pos[1], pos[2]]
        # except IndexError:
        #     agent_ptr.alive = False
        #     agent_ptr.position_history = agent_ptr.get_position_history()[:-2]
        #     return np.zeros(3)

        if mem_or_not > 0:
            agent_ptr.alive = False
            agent_ptr.position_history = agent_ptr.get_position_history()[:-2]
            if self.respawn_on_hit:
                swarm_ptr.add_random(agent_ptr)
        return self.vec


class AdjacencySensor(Sensor):
    """

    Sensor that queries the environment and returns the direction away from other agent paths

    """

    def __init__(self, exp_mat, radius, kill_on_exp=False, history_range=5):
        self.vis = radius
        self.data = exp_mat
        self.kill_on_exp = kill_on_exp
        self.history_range = history_range

    def get_vector(self, swarm_ptr, agent_ptr):
        """
        Get required data for this sensor. Has access to full swarm.

        Arguments:
            swarm_ptr (agents.Swarm): A pointer to the swarm to which this
                sensor's agent belongs
            agent_ptr (agents.Agent): A pointer to the agent

        Returns:
            np.array: The new direction

        """
        position = agent_ptr.position.astype(int)
        # Maybe don't set membranes as explored by default
        # Reason 1 - seperation of concerns - membranes as membranes, explored as explored
        # look at both seperate, perhaps. Look at both scenarios and test
        sub_mat = self.data[
            position[0] - self.vis : position[0] + self.vis + 1,
            position[1] - self.vis : position[1] + self.vis + 1,
            position[2] - self.vis : position[2] + self.vis + 1,
        ]

        # Second term here doesn't work yet. Intent is to kill the agent when it sees another agent
        # But we need to ensure that the agent doesn't kill itself on its own trail behind it. TODO

        return resultant_vector


class PrecomputedSensor(Sensor):
    """
    Based on convolution of the membrane data, query the lookup table
    """

    def __init__(self):
        self.vector = np.zeros((3,))

    def get_vector(self, swarm_ptr, agent_ptr):
        """
        Get required data for this sensor. Has access to full swarm.

        Arguments:
            swarm_ptr (agents.Swarm): A pointer to the swarm to which this
                sensor's agent belongs
            agent_ptr (agents.Agent): A pointer to the agent

        Returns:
            np.array: The new direction

        """
        pos = agent_ptr.get_position()
        # Agent position must be int for array indexing
        pos = pos.astype(int)
        try:
            vec = swarm_ptr.data[tuple(pos)]
        except IndexError:
            agent_ptr.alive = False
            agent_ptr.position_history = agent_ptr.position_history[:-2]
            vec = np.zeros(3)
        # Dealing with anisotropy and row/col vs x/y flip
        print("S", pos, vec.shape)
        vec[0], vec[1], vec[2] = vec[1], vec[0], vec[2]
        self.vector = vec
        return self.vector


class CrossSensor(Sensor):
    """
    Finds vector orthogonal to precomputed vector to weakly follow walls
    This is similar to the cross product, but it uses the agent's velocity
    as context
    """

    def __init__(self, use_yz=False):
        self.vector = np.zeros((3,))
        self.pos = 0
        self.use_yz = use_yz

    # want to spawn half of the agents with one cross prod, half with the other
    def get_vector(self, swarm_ptr, agent_ptr):
        """
        Get required data for this sensor. Has access to full swarm.

        Arguments:
            swarm_ptr (agents.Swarm): A pointer to the swarm to which this
                sensor's agent belongs
            agent_ptr (agents.Agent): A pointer to the agent

        Returns:
            np.array: The new direction

        """
        pos = agent_ptr.get_position()
        vel = agent_ptr.velocity
        # Agent position must be int for array indexing
        pos = pos.astype(int)
        self.pos = pos
        try:
            vec = swarm_ptr.data[tuple(pos)]
        except IndexError:
            agent_ptr.alive = False
            return np.zeros(3)
        # Dealing with anisotropy and row/col vs x/y flip
        vec[0], vec[1], vec[2] = vec[1], vec[0], vec[2] / swarm_ptr.isotropy
        # Comparing the direction of two vectors so that we add the right
        # Direction of the cross product -> this pushes the agents along the
        # Wall in the x/y plane in the same direction it is moving.

        if vel[1] * vec[0] > vel[0] * vec[1]:
            vec_perp = np.array([-vec[1], vec[0], 0])
        else:
            vec_perp = np.array([vec[1], -vec[0], 0])
        # Unstable for anisotropic volumes but useful otherwise
        if self.use_yz:
            if vel[2] * vec[1] > vel[1] * vec[2]:
                vec_perpyz = np.array([0, -vec[2], vec[1]])
            else:
                vec_perpyz = np.array([0, vec[2], -vec[1]])

            if vel[2] * vec[0] > vel[0] * vec[2]:
                vec_perpzx = np.array([vec[2], 0, -vec[0]])
            else:
                vec_perpzx = np.array([-vec[2], 0, vec[0]])
            vec_perp += vec_perpyz + vec_perpzx

        self.vector = vec_perp
        return self.vector


# look at changes for bounching between volumes
# spawn agent in a new swarm
class DataBoundarySensor(Sensor):
    """
    Keeps agent in bounds of volume
    """

    def __init__(self, radius=10):
        self.radius = radius
        self.vector = np.zeros((3,))

    def get_vector(self, swarm_ptr, agent_ptr):
        """
        Get required data for this sensor. Has access to full swarm.

        Arguments:
            swarm_ptr (agents.Swarm): A pointer to the swarm to which this
                sensor's agent belongs
            agent_ptr (agents.Agent): A pointer to the agent

        Returns:
            np.array: The new direction

        """
        self.vector = np.zeros((3,))
        # If agent leaves the volume it dies
        for i in [0, 1, 2]:
            if agent_ptr.position[i] < self.radius:
                self.vector[i] = 1.0
            elif agent_ptr.position[i] > (swarm_ptr.data.shape[i] - self.radius):
                self.vector[i] = -1.0
            elif agent_ptr.position[i] < 0 or agent_ptr.position[i] >= swarm_ptr.data.shape[i] - 1:
                agent_ptr.alive = False
                agent_ptr.position_history = agent_ptr.position_history[:-2]
                swarm_ptr.add_random(agent_ptr)
        return self.vector

# TODO - maximum velocity probing (Using edt?)
