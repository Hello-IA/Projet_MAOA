# a hamiltonian cycle (t1, ..., tn) called tour passing exactly once through each city
# a packing plan encoded by yik item i collected at city i
# repects the W constraint

import numpy as np
import copy
from typing import List, Dict, Set, Tuple, Optional


class TWDTSPSolution:
    """
    Solution representation for the Traveling Thief Problem with Dynamic Time-dependent Speed (TWDTSP):
    - A Hamiltonian cycle passing exactly once through each city
    - A packing plan indicating which items are collected at which cities
    """
    
    def __init__(self, tour: List[int], 
                 packing_plan: Dict[int, Set[int]], 
                 total_profit: float,
                 total_weight: float, 
                 max_weight: float):
        
        self.tour = tour # List of city indices representing the Hamiltonian cycle
        self.packing_plan = packing_plan # Dictionary mapping city_index -> set of item indices collected at that city
        self.total_profit = total_profit
        self.total_weight = total_weight
        self.W = max_weight
        
        # Validate the solution upon creation
        self._validate()
    
    def _validate(self):
        """
        Validate that the solution is well-formed
        """
        # valid Hamiltonian cycle
        if len(self.tour) < 2:
            raise ValueError("Tour must contain at least 2 cities")
        
        # valid cycle
        if self.tour[0] != self.tour[-1]:
            raise ValueError(f"Tour must be a cycle (start and end at same city). Got: {self.tour[0]} != {self.tour[-1]}")
        
        # each city appears once
        unique_cities = set(self.tour[:-1])  # repeated cycle start
        if len(unique_cities) != len(self.tour) - 1:
            raise ValueError("Tour must visit each city exactly once (Hamiltonian cycle)")
        
        # valid city indices
        tour_cities = set(self.tour)
        for city in self.packing_plan.keys():
            if city not in tour_cities:
                raise ValueError(f"Packing plan references city {city} which is not in the tour")
    
    def is_feasible(self) -> bool:
        """
        Check if the solution respects the weight constraint W
        """
        if self.total_profit <= self.W:
            return self
        return None
    
    def num_items_collected(self) -> int:
        """
        Return the total number of items collected
        """
        return sum(len(item_set) for item_set in self.packing_plan.values())
    
    def get_collected_items(self) -> List[Tuple[int, int]]:
        """
        Get a list of all collected items as (city, item_index) tuples
        """
        collected = []
        for city, item_indices in self.packing_plan.items():
            for item_idx in item_indices:
                collected.append((city, item_idx))
        return collected
    
    def add_item(self, city: int, item_idx: int, item_profit: float, item_weight: float):
        """
        Add an item to the packing plan and returns the new solution
        """
        solution = copy.deepcopy(self)
        
        if city not in solution.packing_plan:
            solution.packing_plan[city] = set()
        solution.packing_plan[city].add(item_idx)
        solution.total_profit += item_profit
        solution.total_weight += item_weight
        
        return solution
        
    
    def remove_item(self, city: int, item_idx: int):
        """
        Remove an item from the packing plan.
        """
        solution = copy.deepcopy(self)

        if city in solution.packing_plan and item_idx in solution.packing_plan[city]:
            solution.packing_plan[city].remove(item_idx)
            if len(solution.packing_plan[city]) == 0:
                solution.packing_plan.pop(city, None)
        
        return solution
    
    def __repr__(self) -> str:
        tour_str = " => ".join(map(str, self.tour[:5])) + ("..." if len(self.tour) > 5 else "")
        num_items = self.num_items_collected()
        return f"TWDTSPSolution(tour_length={len(self.tour)-1}, items_collected={num_items}, W={self.W})"
    
    def __str__(self) -> str:
        return (f"TWDTSP Solution:\n"
                f"  Tour: {self.tour}\n"
                f"  Items collected: {self.num_items_collected()}\n"
                f"  Total weight: {self.total_weight}\n"
                f"  Total profit: {self.total_profit}\n"
                f"  Max weight: {self.W}")


# Example usage
if __name__ == "__main__":
    # Example: 4 cities tour (0 → 1 → 2 → 3 → 0)
    tour = [0, 1, 2, 3, 0]
    
    # Example: Collect items at cities 1 and 3
    # At city 1: collect items with indices 0 and 2
    # At city 3: collect item with index 1
    packing_plan = {
        1: {0, 2},
        3: {1}
    }
    
    max_weight = 100.0
    
    # Create solution
    solution = TWDTSPSolution(tour, packing_plan, max_weight)
    print(solution)
    print(f"\nItems collected: {solution.get_collected_items()}")
    
    # Example items data structure (city -> [profit, weight] pairs)
    items = {
        1: np.array([[10.0, 5.0], [15.0, 8.0], [12.0, 6.0]]),  # 3 items at city 1
        2: np.array([[8.0, 4.0]]),                              # 1 item at city 2
        3: np.array([[20.0, 10.0], [18.0, 9.0]])               # 2 items at city 3
    }
    
    print(f"\nTotal weight: {solution.total_weight(items):.2f}")
    print(f"Total profit: {solution.total_profit(items):.2f}")
    print(f"Is feasible: {solution.is_feasible(items)}")
    
    # Modify solution
    solution.add_item(2, 0)  # Add item 0 from city 2
    print(f"\nAfter adding item:")
    print(f"Total weight: {solution.total_weight(items):.2f}")
    print(f"Total profit: {solution.total_profit(items):.2f}")
    
    # Create a copy
    solution_copy = solution.copy()
    solution_copy.remove_item(1, 0)
    print(f"\nCopied solution after removing item:")
    print(f"Items in copy: {solution_copy.num_items_collected()}")
    print(f"Items in original: {solution.num_items_collected()}")