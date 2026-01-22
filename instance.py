from output import TWDTSPSolution

import numpy as np
from typing import Dict


class TWDTSP:
    """
    Time and Weight Dependent Travelling Salesman Problem
    """

    def __init__(
        self,
        coords: np.ndarray,             # shape (n, 2)
        items: Dict[int, np.ndarray],   # city -> [[profit, weight], ...]
        max_weight: float,
        edge_weight_type: str = "CEIL_2D",
    ):
        self.coords = coords
        self.m = items
        self.W = max_weight
        self.n = coords.shape[0]
        self.edge_weight_type = edge_weight_type

        self.max_items = max(v.shape[0] for v in items.values())

        self.p = np.zeros((self.n, self.max_items))
        self.w = np.zeros((self.n, self.max_items))

        for city, arr in items.items():
            n = arr.shape[0]
            self.p[city - 1, :n] = arr[:, 0]
            self.w[city - 1, :n] = arr[:, 1]


    def city_items(self, city: int) -> np.ndarray:
        return self.m.get(city, np.empty((0, 2)))

    def distance(self, i: int, j: int) -> float: # indexing is done as in the ttp format starting from 1
        """
        Compute distance on demand (1-based indexing).
        """
        dx = self.coords[i - 1, 0] - self.coords[j - 1, 0]
        dy = self.coords[i - 1, 1] - self.coords[j - 1, 1]
        dist = np.sqrt(dx * dx + dy * dy)

        if self.edge_weight_type == "CEIL_2D":
            return np.ceil(dist)

        return dist
    
    def objective_function(self):
        """
        Outputs objective function according to a transport cost function
        """
        def obj(solution: TWDTSPSolution):
        
            total_profit = 0
            for i in range(self.n):
                total_profit += np.sum(solution.packing_plan[i, :] * self.p[i])
        
            total_cost = 0
            wi = 0
            for i in range(self.n - 1):
                city = solution.tour[i]
                next_city = solution.tour[i + 1]
                wi += np.sum(solution.packing_plan[city - 1, :] * self.w[city - 1])
                total_cost += self.transport_cost(city, next_city, wi)
            total_cost += self.transport_cost(solution.tour[-1], solution.tour[0], wi)
        
            return total_profit - total_cost
        return obj
    
    def transport_cost(self, i: int, j: int, carried_weight: float) -> float:
        raise NotImplementedError("Subclasses must implement transport_cost")
    
    def as_kctsp(self, weight_cost_per_km: float) -> 'KCTSPProblem':
        """
        Formulate the instance as a KCTSP problem
        """
        return KCTSPProblem(self.coords, self.m, self.W, weight_cost_per_km, self.edge_weight_type)
    
    def as_ttp(self, min_speed: float, max_speed:float, transport_cost_per_second: float) -> 'TTPProblem':
        """
        Formulate the instance as a TTP problem
        """
        return TTPProblem(self.coords, self.m, self.W, min_speed, max_speed, transport_cost_per_second, self.edge_weight_type)


class KCTSPProblem(TWDTSP):
    """
    Knapsack-Constrained TSP with cumulative weight cost
    """

    def __init__(
        self,
        coords: np.ndarray,             # shape (n, 2)
        items: Dict[int, np.ndarray],   # city -> [[profit, weight], ...]
        max_weight: float,
        weight_cost_per_km: float,
        edge_weight_type: str = "CEIL_2D",
    ):
        super().__init__(coords, items, max_weight, edge_weight_type)
        self.kw = weight_cost_per_km

    def transport_cost(self, i: int, j: int, carried_weight: float) -> float:
        """
        Cost of transporting the current load over the distance between city i and j
        """
        dist = self.distance(i, j)
        return self.kw * carried_weight * dist


class TTPProblem(TWDTSP):
    """
    Traveling Thief Problem variant with speed and time-based cost
    """

    def __init__(
        self,
        coords: np.ndarray,             # shape (n, 2)
        items: Dict[int, np.ndarray],   # city -> [[profit, weight], ...]
        max_weight: float,
        min_speed: float,
        max_speed: float,
        transport_cost_per_second: float,
        edge_weight_type: str = "CEIL_2D",
    ):
        super().__init__(coords, items, max_weight, edge_weight_type)
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.kd = transport_cost_per_second

    def travel_time(self, i: int, j: int, carried_weight: float) -> float:
        """
        Time needed to travel from city i to j at given speed
        """
        dist = self.distance(i, j)
        v = (self.max_speed - self.min_speed) / self.W
        return dist / (self.max_speed - v * carried_weight)

    def transport_cost(self, i: int, j: int, carried_weight: float) -> float:
        """
        Cost of transport based on travel time between cities i and j
        """
        time = self.travel_time(i, j, carried_weight)
        return self.kd * time