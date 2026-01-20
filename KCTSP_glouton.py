from utils_io import TWDTSPLoader
from instance import KCTSPProblem
import pandas as pd
import numpy as np
import time
from utils_KCTSP import *
import math






# ======================================================
# 4) Solveur KCTSP glouton complet
# ======================================================
def solve_kctsp_greedy(problem, start_city=1):
    tour = tsp_nearest_neighbor(problem, start=start_city)
    start = time.time()
    
    packing_plan, profit = greedy_kctsp_knapsack(problem, tour)
    end = time.time()
    print("time greedy_kctsp_knapsack :", end-start)
    return {
        "tour": tour,
        "packing_plan": packing_plan,
        "profit": profit
    }
    

    
def solve_kctsp_exact(problem, start_city=1):
    tour = tsp_nearest_neighbor(problem, start=start_city)
    packing_plan, profit = solve_kctsp_knapsack_exact(problem, tour)

    return {
        "tour": tour,
        "packing_plan": packing_plan,
        "profit": profit
    }




if __name__ == "__main__":
    
    db = pd.DataFrame(columns=[
        'problem_name', 'min_speed', 'max_speed', 'renting_ratio',
        'knapsack_data_type', 'dimension', 'num_items', 'max_weight', 'edge_weight_type'
    ])
    
    # Load a single file without populating database
    problem_tw = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n4460_bounded-strongly-corr_01.ttp")

    kw = 1.0/problem_tw.n  # coût par km et par kg (à choisir)
    problem_kc = problem_tw.as_kctsp(weight_cost_per_km=kw)
    print("solve_kctsp_greedy", solve_kctsp_greedy(problem_kc))
    #print("solve_kctsp_exact", solve_kctsp_exact(problem_kc))
    


        
