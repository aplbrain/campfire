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
from typing import List, Tuple, Dict

import numpy as np

from agents.sensor import Sensor


class Agent:
    """
    Base class for an agents.

    Includes logic for communication with swarm, etc.

    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (0, 0, 0),
        agent_id: int=-1,
        seg_id: int=-1,
        sensors: List[Tuple[Sensor, float]] = None,
        **kwargs
    ) -> None:
        """
        Create a new agent.

        Arguments:
            sensors (Dict[Sensor, float]): The sensors to control the agent.
                The key is the sensor object, and the value is the weight.

        """
        # Default to alive; but dead agents CAN be created (possibly useful
        # when writing avoid-peers sensors).

        self.position = np.array(position, dtype=np.float)  # type: np.array
        self.velocity = np.array((0.0, 0.0, 0.0), dtype=np.float)  # type: np.array
        self.alive = kwargs.get("alive", True)
        self.max_velocity = kwargs.get("max_velocity", 1.0)
        self._position_history: List[np.array] = []
        self._position_history.append(np.copy(self.position))
        self.seg_id = seg_id
        self.agent_id = agent_id

        if sensors is not None:
            if not isinstance(sensors, list):
                raise TypeError(
                    "`sensors` argument must be of type " "List[Tuple[Sensor, float]]"
                )

            if not isinstance(sensors[0], tuple):
                raise TypeError(
                    "`sensors` argument must be of type " "List[Tuple[Sensor, float]]"
                )

            for sensor, _ in sensors:
                if (
                    not isinstance(sensor, Sensor)  # or
                    # (not isinstance(weight, float))
                ):
                    raise TypeError(
                        "`sensors` argument must be of type "
                        "List[Tuple[Sensor, float]]"
                    )
            self.sensors = sensors  # type: List[Tuple[Sensor, float]]

    def step(self, swarm_ptr) -> bool:
        """
        Step one iteration in the simulation.

        Arguments:
            swarm_ptr (agents.Swarm): The swarm to which this agent belongs.
                Used for data-polling etc.

        Returns:
            bool: True if still alive at the end of this step.

        """
        # Loop through sensors, averaging each (weighted) with all
        # existing measurements
        acceleration = np.zeros((3,))
        for (sensor, weight) in self.sensors:
            vec = sensor.get_vector(swarm_ptr, self)
            # print(sensor, vec.shape)
            acceleration += weight * vec

        # Set z velocity lower as function of anisotropy
        self.velocity += [acceleration[0], acceleration[1], acceleration[2] / (swarm_ptr.isotropy)]
        if np.linalg.norm(self.velocity) >= self.max_velocity:
            self.velocity = (
                self.velocity / np.linalg.norm(self.velocity) * self.max_velocity
            )

        self.position += self.velocity
        self._position_history.append(np.copy(self.position))
        return True

    def get_position(self) -> np.array:
        """
        Return the position of the agent at the current timestamp.

        Arguments:
            None

        Returns:
            numpy.array: The 3D position vector

        """
        return self.position

    def get_position_history(self) -> List[np.array]:
        """
        Get the position history of the agent.

        Arguments:
            None

        Returns:
            List[np.array]: a list of (3, n)-shaped numpy arrays, where each
                item is an XYZ tuple of the agent's position at that time

        """
        return self._position_history
