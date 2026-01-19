import gurobipy as gp
from gurobipy import GRB
import numpy as np

def tsp_nearest_neighbor(problem, start=1):
    n = problem.n
    visited = set([start])
    tour = [start]
    current = start

    while len(visited) < n:
        next_city = None
        best_dist = float("inf")

        for j in range(1, n + 1):
            if j in visited:
                continue
            d = problem.distance(current, j)
            if d < best_dist:
                best_dist = d
                next_city = j

        tour.append(next_city)
        visited.add(next_city)
        current = next_city

    tour.append(start)  # retour au dépôt
    return tour


# ======================================================
# 2) Distances restantes
# ======================================================
def compute_remaining_distances(problem, tour):
    remaining = {}
    total_dist = 0.0

    # distance totale
    for i in range(len(tour) - 1):
        total_dist += problem.distance(tour[i], tour[i + 1])

    current_dist = total_dist
    for i in range(len(tour) - 1):
        city = tour[i]
        remaining[city] = current_dist
        current_dist -= problem.distance(tour[i], tour[i + 1])

    return remaining


# ======================================================
# 3) Knapsack glouton KCTSP
# ======================================================
def greedy_kctsp_knapsack(problem, tour):
    remaining_dist = compute_remaining_distances(problem, tour)

    candidates = []

    # construction des objets candidats
    for city in tour:
        for j, (profit, weight) in enumerate(problem.city_items(city)):
            transport_cost = problem.kw * weight * remaining_dist[city]
            net_profit = profit - transport_cost

            if net_profit > 0:
                ratio = net_profit / weight
                candidates.append((ratio, city, j, profit, weight))

    # tri glouton
    candidates.sort(reverse=True)

    capacity = problem.W
    packing_plan = {}
    total_profit = 0.0

    for _, city, j, profit, weight in candidates:
        if weight <= capacity:
            capacity -= weight
            total_profit += (profit - problem.kw * weight * remaining_dist[city])
            packing_plan.setdefault(city, []).append(j)

    return packing_plan, total_profit



def compute_remaining_distances(problem, tour):
    """
    Calcule la distance restante après chaque ville du tour.
    
    Args:
        problem : KCTSPProblem
        tour : list[int] (1-based)
        
    Returns:
        remaining_dist : dict[int, float]
            distance restante à partir de chaque ville
    """
    n = len(tour)
    remaining_dist = {}

    for i in range(n):
        d = 0.0
        for k in range(i, n - 1):
            d += problem.distance(tour[k], tour[k + 1])
        # retour au dépôt
        d += problem.distance(tour[-1], tour[0])
        remaining_dist[tour[i]] = d

    return remaining_dist

def solve_kctsp_knapsack_exact(problem, tour, time_limit=30, verbose=True):
    """
    Résout exactement le KCTSP pour un tour fixé via PLNE (Gurobi).
    
    Args:
        problem : KCTSPProblem
        tour : list[int] (1-based)
        time_limit : int (secondes)
        verbose : bool
        
    Returns:
        packing_plan : dict[int, list[int]]
        objective_value : float
    """
    remaining_dist = compute_remaining_distances(problem, tour)

    m = gp.Model("KCTSP_Knapsack_FixedTour")

    if not verbose:
        m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    y = {}   # variable par objet
    obj_terms = []

    # création des variables
    for city in tour:
        items = problem.city_items(city)
        if len(items) == 0:
            continue

        for j, (profit, weight) in enumerate(items):
            var_name = f"y_{city}_{j}"
            y[(city, j)] = m.addVar(vtype=GRB.BINARY, name=var_name)

            net_profit = profit - problem.kw * weight * remaining_dist[city]
            obj_terms.append(net_profit * y[(city, j)])

    # fonction objectif
    m.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)

    # contrainte de capacité
    m.addConstr(
        gp.quicksum(
            problem.city_items(city)[j][1] * y[(city, j)]
            for (city, j) in y
        ) <= problem.W,
        name="capacity"
    )

    m.optimize()

    # extraction de la solution
    packing_plan = {}
    for (city, j), var in y.items():
        if var.X > 0.5:
            packing_plan.setdefault(city, []).append(j)

    return packing_plan, m.ObjVal