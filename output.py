from graphviz import Digraph
import numpy as np
import copy
from typing import List, Dict, Set, Tuple, Optional


class TWDTSPSolution:
    """
    Solution representation for the Traveling Thief Problem with Dynamic Time-dependent Speed (TWDTSP):
    - A Hamiltonian cycle passing exactly once through each city
    - A binary packing plan (numpy array) where packing_plan[city][item] = 1 if item is taken, 0 otherwise
    """
    
    def __init__(self, tour: List[int], 
                 packing_plan: np.ndarray,  # a 2D binary array: [num_cities x max_items_per_city]
                 total_profit: float,
                 total_weight: float, 
                 max_weight: float):
        
        self.tour = tour  # List of city indices representing the Hamiltonian cycle
        self.packing_plan = packing_plan  # Binary array indicating which items are collected
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
        unique_cities = set(self.tour[:-1])  # exclude repeated cycle start
        if len(unique_cities) != len(self.tour) - 1:
            raise ValueError("Tour must visit each city exactly once (Hamiltonian cycle)")
        
        # validate packing plan is binary
        if not np.all((self.packing_plan == 0) | (self.packing_plan == 1)):
            raise ValueError("Packing plan must be binary (only 0s and 1s)")
    
    def is_feasible(self) -> bool:
        """
        Check if the solution respects the weight constraint W
        """
        return self.total_weight <= self.W
    
    def num_items_collected(self) -> int:
        """
        Return the total number of items collected
        """
        return np.sum(self.packing_plan)
    
    def get_collected_items(self, item_details) -> List[Tuple[int, int]]:
        """
        Get a list of all collected items as (city, item_index) tuples
        """
        collected = []
        cities, items = np.where(self.packing_plan == 1)
        for city, item_idx in zip(cities, items):
            if city in item_details.keys():
                profit, weight = item_details[city][int(item_idx)]
                collected.append((int(city), int(item_idx), profit, weight))
        return collected
    
    def add_item(self, city: int, item_idx: int, item_profit: float, item_weight: float):
        """
        Add an item to the packing plan and returns the new solution
        """
        solution = copy.deepcopy(self)
        
        if solution.packing_plan[city - 1, item_idx] == 0:
            solution.packing_plan[city - 1, item_idx] = 1
            solution.total_profit += item_profit
            solution.total_weight += item_weight
        
        return solution
    
    def remove_item(self, city: int, item_idx: int, item_profit: float, item_weight: float):
        """
        Remove an item from the packing plan.
        """
        solution = copy.deepcopy(self)

        if solution.packing_plan[city, item_idx] == 1:
            solution.packing_plan[city, item_idx] = 0
            solution.total_profit -= item_profit
            solution.total_weight -= item_weight
        
        return solution
    
    def __repr__(self) -> str:
        tour_str = " => ".join(map(str, self.tour[:5])) + ("..." if len(self.tour) > 5 else "")
        num_items = self.num_items_collected()
        return f"TWDTSPSolution(tour_length={len(self.tour)-1}, items_collected={num_items}, W={self.W})"
    
    def __str__(self) -> str:
        return (f"TWDTSP Solution:\n"
                f"  Tour: {self.tour}\n"
                f"  Items collected: {self.num_items_collected()}\n"
                f"  Total weight: {self.total_weight:.2f}\n"
                f"  Total profit: {self.total_profit:.2f}\n"
                f"  Max weight: {self.W}")
        
    def display(self, coords: np.ndarray, items: Dict[int, np.ndarray], scale:float=5, padding:float=0.5, path:str=None):
        # initializing the graph
        dot = Digraph(
            comment="TWDTSP Solution",
            engine="neato"
            )
        
        dot.graph_attr.update({
            "pad": f"{padding},{padding}"  # left/right and top/bottom padding
            })
        
        # normalizing coordinates
        coords = coords.astype(float, copy=True)
        mins = coords.min(axis=0)
        ranges = (coords.max(axis=0) - mins) # the margin is an innner margin 
        graph_scale = np.max(ranges)
        if graph_scale > 0:
            coords -= mins  # avoiding blank space on the left
            coords /= graph_scale # normalization
                
        # color mapping function for the edges
        def set_color(w):
            t = max(0.0, min(w, self.W)) / self.W
            R = int(255 * t)
            G = 0
            B = int(255 * (1 - t))
            return f"#{R:02x}{G:02x}{B:02x}"
    
        # calculate profit trace per city
        city_profits = {}
        profit = 0
        for city in self.tour:
            if city < self.packing_plan.shape[0]:
                # Sum profits of all items taken at this city
                for item_idx in range(self.packing_plan.shape[1]):
                    if self.packing_plan[city, item_idx] == 1:
                        if city in items and item_idx < len(items[city]):
                            profit += items[city][item_idx, 0]
            city_profits[city] = profit
        
        # draw city nodes 
        for city in range(coords.shape[0]):
            x, y = coords[city]
            label = str(city)  
            dot.node(
                str(city),
                label=label,
                pos=f"{x * scale},{y * scale}!",
                shape="circle",
                width=f"{scale * 0.3 / 5}",  # 0.3 at scale=5
                fixedsize="true"
                )
        
        # drawing tour edges with weight coloration
        weight = 0
        for i in range(len(self.tour) - 1):
            A = self.tour[i]
            B = self.tour[i + 1]
            if A < self.packing_plan.shape[0]:
                # Add weight of all items taken at city A
                for item_idx in range(self.packing_plan.shape[1]):
                    if self.packing_plan[A, item_idx] == 1:
                        if A in items and item_idx < len(items[A]):
                            weight += items[A][item_idx, 1]
            dot.edge(
                str(A),
                str(B),
                color=set_color(weight),
                penwidth=f"{scale * 2 / 5}"  # 2 at scale=5
                )
            
        # double circle on starting node of the tour
        start = self.tour[0]
        dot.node(
            str(start),
            shape="doublecircle"
            )
        """
        # add profit labels on cities
        for city in range(coords.shape[0]):
            x, y = coords[city]
            dot.node(
                f"city_legend_label_{city}",
                label=f"p = {city_profits[city]:.1f}",
                pos=f"{x * scale},{y * scale + scale * 0.35 / 5}!",  # 0.35 at scale=5
                shape="box",
                style="filled",
                fillcolor ="#ffffffcc",   # white, 70% opacity
                color="black",
                fontcolor="black",
                fontsize=f"{scale * 7 / 5}",  # 7 at scale=5
                margin=f"{scale * 0.001 / 5}",  # 0.001 at scale=5
                width=f"{scale * 0.8 / 5}",  # 0.8 at scale=5
                height=f"{scale * 0.2 / 5}",  # 0.2 at scale=5
                fixedsize="false"
            )
        """
        # legend for weight coloration
        LEGEND_X = scale + scale * 1 / 5  # (scale + 1) at scale=5
        N_LEGEND = 30      # number of gradient steps
        for i in range(N_LEGEND):
            w = self.W * i / (N_LEGEND - 1)
            y_pos = scale * i / (N_LEGEND - 1)  # vertical position
            dot.node(
                f"legend_{i}",
                label=f"",
                shape="box",
                width=f"{scale * 0.5 / 5}",  # 0.5 at scale=5
                height=str(float(scale) / float(N_LEGEND)),
                fixedsize="true",
                style="filled",
                fillcolor=set_color(w),
                color = set_color(w),
                pos=f"{LEGEND_X},{y_pos}!"
            )
            dot.node(
                f"legend_label_{i}",
                label=f"{w:.2f}",
                shape="plaintext",
                fontsize=f"{scale * 8 / 5}",  # 8 at scale=5
                pos=f"{LEGEND_X + scale * 0.6 / 5},{y_pos - float(scale) / float(N_LEGEND) * 0.5}!"  # 0.6 at scale=5
                )
        
        # ---------- render ----------
        if not path:
            path = "solution"
        dot.render(path, format="png", cleanup=True)


# Temporary helper function to convert old dict-based packing plan to binary array
def dict_to_binary_packing(packing_dict: Dict[int, Set[int]], 
                           num_cities: int, 
                           max_items_per_city: int) -> np.ndarray:
    """
    Convert a dictionary-based packing plan to a binary numpy array
    """
    packing_array = np.zeros((num_cities, max_items_per_city), dtype=int)
    for city, item_set in packing_dict.items():
        for item_idx in item_set:
            packing_array[city, item_idx] = 1
    return packing_array


# Example usage
if __name__ == "__main__":
    # Example: 4 cities tour (0 → 1 → 2 → 3 → 0)
    tour = [0, 1, 2, 3, 0]
    
    # Example items data structure (city -> [profit, weight] pairs)
    items = {
        1: np.array([[10.0, 5.0], [15.0, 8.0], [12.0, 6.0]]),  # 3 items at city 1
        2: np.array([[8.0, 4.0]]),                              # 1 item at city 2
        3: np.array([[20.0, 10.0], [18.0, 9.0]])               # 2 items at city 3
    }
    
    # Create binary packing plan
    # Shape: [num_cities, max_items]
    num_cities = 4
    max_items = 3  # Maximum items at any city
    packing_plan = np.zeros((num_cities, max_items), dtype=int)
    
    # At city 1: collect items with indices 0 and 2
    packing_plan[1, 0] = 1
    packing_plan[1, 2] = 1
    
    # At city 3: collect item with index 1
    packing_plan[3, 1] = 1
    
    max_weight = 25.0
    
    # Calculate initial totals
    total_profit = items[1][0, 0] + items[1][2, 0] + items[3][1, 0]
    total_weight = items[1][0, 1] + items[1][2, 1] + items[3][1, 1]
    
    # Create solution
    solution = TWDTSPSolution(tour, packing_plan, total_profit, total_weight, max_weight)
    print(solution)
    print(f"\nItems collected: {solution.get_collected_items(items)}")
    
    print(f"\nTotal weight: {solution.total_weight:.2f}")
    print(f"Total profit: {solution.total_profit:.2f}")
    print(f"Is feasible: {solution.is_feasible()}")
    
    # Modify solution
    solution = solution.add_item(2, 0, items[2][0, 0], items[2][0, 1])  # Add item 0 from city 2
    print(f"\nAfter adding item:")
    print(f"Total weight: {solution.total_weight:.2f}")
    print(f"Total profit: {solution.total_profit:.2f}")
    
    # Create a copy
    solution_copy = copy.deepcopy(solution)
    solution_copy = solution_copy.remove_item(1, 0, items[1][0, 0], items[1][0, 1])
    print(f"\nCopied solution after removing item:")
    print(f"Items in copy: {solution_copy.num_items_collected()}")
    print(f"Items in original: {solution.num_items_collected()}")
    
    # Display coordinates
    coords = np.array([
        [10.0, 50.0],   # city 0
        [30.0, 20.0],   # city 1
        [80.0, 90.0],   # city 2
        [60.0, 10.0],   # city 3
    ])
    
    solution.display(coords, items)