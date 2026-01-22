from instance import TTPProblem

import numpy as np
from typing import List

def insertion_heuristic(problem: TTPProblem, tour: List[int]) -> np.ndarray:
        """
        Insertion heuristic based on Algorithm 2 from the paper
        Uses three approximations: empty-tour, worst-case, and expected increased time
        """
        max_items = problem.p.shape[1]
        packing_plan = np.zeros((problem.n, max_items), dtype=int)
        W = 0.0  # Current total weight
        
        # Calculate tour length and location indices
        tour_length = 0.0
        location_idx = {}
        for i in range(len(tour) - 1):
            location_idx[tour[i]] = i
            tour_length += problem.distance(tour[i], tour[i + 1])
        
        # Calculate remaining distance from each city to tour end
        remaining_distance = {}
        for i in range(len(tour) - 1, 0, -1):
            if i == len(tour) - 1:
                remaining_distance[tour[i - 1]] = problem.distance(tour[i - 1], tour[i])
            else:
                remaining_distance[tour[i - 1]] = (
                    remaining_distance[tour[i]] + 
                    problem.distance(tour[i - 1], tour[i])
                )
        
        # Create item list with priorities based on empty-tour approximation (Eq. 22)
        items_list = []
        vmax = problem.max_speed
        vmin = problem.min_speed
        nu = (vmax - vmin) / problem.W
        R = problem.kd
        
        for city_idx in range(problem.n):
            city = city_idx + 1
            if city not in remaining_distance:
                continue
            
            L_loc = remaining_distance[city]
            
            for item_idx in range(max_items):
                profit = problem.p[city_idx, item_idx]
                weight = problem.w[city_idx, item_idx]
                
                if weight > 0:
                    # Empty-tour increased time (Eq. 22)
                    delta_t1 = L_loc * (1 / (vmax - nu * weight) - 1 / vmax)
                    priority = (profit - R * delta_t1) / weight
                    items_list.append((priority, city_idx, item_idx, profit, weight))
        
        # Sort by priority (descending)
        items_list.sort(reverse=True, key=lambda x: x[0])
        
        # Insert items
        for priority, city_idx, item_idx, profit, weight in items_list:
            if W + weight <= problem.W:
                city = city_idx + 1
                L_loc = remaining_distance.get(city, 0)
                
                # Worst-case increased time (Eq. 24)
                if W + weight > 0:
                    delta_t2 = L_loc * (
                        1 / (vmax - nu * (W + weight)) - 
                        1 / (vmax - nu * W)
                    )
                else:
                    delta_t2 = 0
                
                # Check worst-case approximation
                if profit > R * delta_t2:
                    packing_plan[city_idx, item_idx] = 1
                    W += weight
                else:
                    # Expected increased time (Eq. 27)
                    if tour_length > 0 and W > 0:
                        a = W / tour_length
                        b1 = vmax - nu * (W + weight)
                        b2 = vmax - nu * W
                        
                        if b1 > 0 and b2 > 0:
                            L_after = tour_length - L_loc
                            delta_t3 = (1 / a) * np.log(
                                ((a * tour_length + b1) * (L_after + b2)) /
                                ((L_after + b1) * (a * tour_length + b2))
                            )
                        else:
                            delta_t3 = delta_t2
                    else:
                        delta_t3 = delta_t2
                    
                    # Check expected approximation
                    if profit > R * delta_t3:
                        packing_plan[city_idx, item_idx] = 1
                        W += weight
        
        return packing_plan