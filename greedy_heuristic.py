from instance import TTPProblem
from output import TWDTSPSolution
from utils_io import TWDTSPLoader
import LK

import random
import numpy as np
import time

class GreedyTTPSolver:
    """
    A TSP-first Greedy two steps algorithm for the Travelling Thief Problem
    """
    def __init__(self, problem: TTPProblem):
        self.problem = problem

    
    def _solve_kp(self, items):
        """
        Greedy method for the KP problem (profit/weight ratio based)
        """
        if not items:
            return []

        W = self.problem.W
        n = len(items)

        # Add index to keep track of original positions
        indexed_items = [
            (i, profit, weight, profit / weight)
            for i, (profit, weight) in enumerate(items)
            if weight > 0
        ]

        # Sort by profit/weight ratio (descending)
        indexed_items.sort(key=lambda x: x[3], reverse=True)

        selected = [0] * n
        remaining_weight = W

        # Greedy selection
        for i, profit, weight, _ in indexed_items:
            if weight <= remaining_weight:
                selected[i] = 1
                remaining_weight -= weight

        return selected


    # TSP first greedy heuristic
    def tsp_first(self, group_size:int=0, picking_factor=0.7, seed=42, tour=None):
        """
        The TTP greedy algorithm
        """
        t0 = time.perf_counter()
        # approximated lower bound for the TTP problem with no items
        if not tour:
            lkSolver = LK.LKSolver(self.problem.coords)
            tour = lkSolver.lk(max_iter=3)
        else:
            tour.pop()
        
        # select groups of knapsack problems
        knapsack_problems = [(city, self.problem.m[city]) for city in tour if city in self.problem.m.keys()]
        if group_size > 0:
            grouped_knapsack_problems = []
            i = 0
            while i < len(knapsack_problems):
                group = []
                for j in range(min(group_size, len(knapsack_problems) - i)):
                    city, items = knapsack_problems[i]
                    group.extend([(city, i, items[i]) for i in range(len(items))])
                    i += 1
                grouped_knapsack_problems.append(group)
            knapsack_problems = grouped_knapsack_problems
        
        # solve knapsack per group
        solutions = []
        for knapsack_problem in knapsack_problems:
            kp = [item for (_, _, item) in knapsack_problem]
            filter = self._solve_kp(kp)
            solution = [knapsack_problem[i] for i in range(len(filter)) if filter[i] == 1]
            solutions.append(solution)

        # select elements from groups with a factor
        # last group has most elements from the solution taken, before last has less, and so on
        solutions.reverse()
        i = 0
        candidates = []
        anti_candidates = []
        random.seed(seed)
        for solution in solutions:
            pick_num = int(picking_factor * (1/(i+1)) * len(solution))
            picked = random.sample(solution, pick_num)
            not_picked = [x for x in solution if x not in picked]
            candidates.extend(picked)
            anti_candidates.extend(not_picked)

        # pick from the candidates to build the packing plan
        packing_plan = np.zeros((self.problem.n, self.problem.max_items))
        tour.append(tour[0])
        ttp_solution = TWDTSPSolution(tour=tour, packing_plan=packing_plan, total_profit=0.0, total_weight=0.0, max_weight=self.problem.W)
        ttp_solution_score = self.problem.objective_function()(ttp_solution)

        for candidate in candidates:
            city, item_idx, item = candidate
            new_ttp_solution = ttp_solution.add_item(city, item_idx, item[0], item[1])
            if new_ttp_solution.is_feasible:
                ttp_solution = new_ttp_solution

        for anti_candidate in anti_candidates:
            city, item_idx, item = anti_candidate
            new_ttp_solution = ttp_solution.add_item(city, item_idx, item[0], item[1])
            new_ttp_solution_score = self.problem.objective_function()(new_ttp_solution)
            if new_ttp_solution_score > ttp_solution_score:
                ttp_solution = new_ttp_solution
                ttp_solution_score = new_ttp_solution_score

        print("score= ", ttp_solution_score)
        print("Total runtime = ", time.perf_counter() - t0)
        return ttp_solution
    
# Example usage
if __name__ == "__main__":

    problem = TWDTSPLoader.load_from_file("./cities280/a280_n2790_uncorr_10.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities280/a280_n279_bounded-strongly-corr_01.ttp", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities280/a280_n1395_uncorr-similar-weights_05.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities280/a280_n2790_uncorr_10.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n4460_bounded-strongly-corr_01.ttp.txt", asTTP=True)

    greedySolver = GreedyTTPSolver(problem)
    
    #solution = greedySolver.tsp_first(group_size=280)
    #solution = greedySolver.tsp_first(group_size=200)
    #solution = greedySolver.tsp_first(group_size=100)
    #solution = greedySolver.tsp_first(group_size=75)
    solution = greedySolver.tsp_first(group_size=50)
    #solution = greedySolver.tsp_first(group_size=25)
    #solution = greedySolver.tsp_first(group_size=10)

    #solution = greedySolver.tsp_first(group_size=50, picking_factor=1)
    #solution = greedySolver.tsp_first(group_size=50, picking_factor=0.75)
    #solution = greedySolver.tsp_first(group_size=50, picking_factor=0.5)
    #solution = greedySolver.tsp_first(group_size=50, picking_factor=0.25)

    print(solution)