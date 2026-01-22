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



def compute_remaining_distances(problem, tour):
    """
    tour : list[int] 1-based et fermé (dernier == premier)
    retourne: dict city(1-based) -> distance restante depuis cette ville (jusqu'au retour dépôt)
    """
    # recuper la liste des cordonner de noeud du graphe
    coords = problem.coords  # shape (n,2)

    # convertir tour (1-based) en indices 0-based
    t = np.asarray(tour, dtype=np.int32) - 1  # shape (L,)
    # arêtes consécutives (je suprime le dernier element de all_less_end et le premeier de all_less_first)
    all_less_end = t[:-1]
    all_less_first = t[1:]
    
    
    #la je souostraire 
    dx = coords[all_less_end, 0] - coords[all_less_first, 0]
    dy = coords[all_less_end, 1] - coords[all_less_first, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    if problem.edge_weight_type == "CEIL_2D":
        dist = np.ceil(dist)
    # dist[k] = distance de tour[k] -> tour[k+1]
    
    #distance de l'ensemble du tour
    total = float(dist.sum())

    # remaining pour chaque position i (sauf dernière, qui est le retour au start)
    # remaining[i] = total - sum(dist[:i])
    #calcule la liste de la some cumuler
    #a = np.array([[1,2,3], [4,5,6]])
    #np.cumsum(a) -> array([ 1,  3,  6, 10, 15, 21])
    prefix = np.concatenate(([0.0], np.cumsum(dist)))  
    #on retire au totale la somme cumuler de 0 jusqua i pour remaining_vals[i] sa lese la somme de i a n-1
    remaining_vals = total - prefix[:-1]               

    # map ville -> remaining (attention: une ville apparaît une seule fois dans un tour TSP)
    remaining = {int(tour[i]): float(remaining_vals[i]) for i in range(len(tour)-1)}
    return remaining


def greedy_kctsp_knapsack(problem, tour):
    
    remaining = compute_remaining_distances(problem, tour)
    kw = problem.kw
    cap = problem.W

    ratios_all = []
    cities_all = []
    idx_all = []
    net_all = []
    w_all = []

    for city in tour[:-1]:
        items = problem.city_items(city)
        if items.size == 0:
            continue

        p = items[:, 0].astype(np.float64, copy=False)
        w = items[:, 1].astype(np.float64, copy=False)
        rem = remaining[city]

        net = p - kw * w * rem
        mask = net > 0
        if not np.any(mask):
            continue

        p2 = p[mask]
        w2 = w[mask]
        net2 = net[mask]
        ratio2 = net2 / w2

        k = len(net2)
        ratios_all.append(ratio2)
        net_all.append(net2)
        w_all.append(w2)
        idx_all.append(np.nonzero(mask)[0].astype(np.int32))
        cities_all.append(np.full(k, city, dtype=np.int32))

    if not ratios_all:
        return {}, 0.0

    ratios = np.concatenate(ratios_all)
    nets = np.concatenate(net_all)
    ws = np.concatenate(w_all)
    idxs = np.concatenate(idx_all)
    cities = np.concatenate(cities_all)

    order = np.argsort(-ratios)  # décroissant

    packing_plan = {}
    obj = 0.0
    for t in order:
        w = float(ws[t])
        if w <= cap:
            cap -= w
            obj += float(nets[t])
            city = int(cities[t])
            j = int(idxs[t])
            packing_plan.setdefault(city, []).append(j)

    return packing_plan, obj

import time

def timed_greedy(problem, tour):
    t0 = time.perf_counter()
    remaining = compute_remaining_distances(problem, tour)
    t1 = time.perf_counter()

    kw = problem.kw
    cap = problem.W

    candidates = []
    for city in tour[:-1]:
        rem = remaining[city]
        items = problem.city_items(city)
        if items.size == 0:
            continue
        p = items[:, 0]
        w = items[:, 1]
        net = p - kw * w * rem
        mask = net > 0
        if mask.any():
            net2 = net[mask]
            w2 = w[mask]
            ratio2 = net2 / w2
            idx2 = mask.nonzero()[0]
            for r, j, nval, ww in zip(ratio2, idx2, net2, w2):
                candidates.append((float(r), city, int(j), float(nval), float(ww)))

    t2 = time.perf_counter()
    candidates.sort(reverse=True)
    t3 = time.perf_counter()

    packing = {}
    obj = 0.0
    for _, city, j, net, w in candidates:
        if w <= cap:
            cap -= w
            obj += net
            packing.setdefault(city, []).append(j)

    t4 = time.perf_counter()
    print(f"remaining: {t1-t0:.3f}s | build candidates: {t2-t1:.3f}s | sort: {t3-t2:.3f}s | fill: {t4-t3:.3f}s | total: {t4-t0:.3f}s")
    return packing, obj




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


import random
import time
from typing import List, Tuple, Dict, Any

def make_random_tour(n: int, start: int = 1) -> List[int]:
    """Tour 1-based fermé."""
    cities = list(range(1, n + 1))
    cities.remove(start)
    random.shuffle(cities)
    return [start] + cities + [start]

def two_opt_apply(tour: List[int], i: int, j: int) -> List[int]:
    """
    2-opt sur un tour fermé.
    i, j sont des indices sur la liste tour (0..len-1).
    On inverse tour[i:j] (segment interne).
    """
    new = tour[:]
    new[i:j] = reversed(new[i:j])
    return new

def improve_tour_2opt_kctsp(
    problem,
    tour: List[int],
    iters: int = 2000,
    time_limit: float = None,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[List[int], float, Dict[int, List[int]]]:
    """
    Recherche locale 2-opt guidée par greedy_kctsp_knapsack.
    """
    random.seed(seed)
    t0 = time.perf_counter()

    best_tour = tour[:]
    best_pack, best_val = greedy_kctsp_knapsack(problem, best_tour)

    n = len(best_tour) - 1  # last is start, nodes indices 0..n
    # indices valides pour 2-opt : on évite de toucher [0] et [-1]
    # donc i in [1, n-2], j in [i+1, n-1]
    for _ in range(iters):
        if time_limit is not None and (time.perf_counter() - t0) > time_limit:
            break

        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)

        cand_tour = two_opt_apply(best_tour, i, j)
        cand_pack, cand_val = greedy_kctsp_knapsack(problem, cand_tour)

        if cand_val > best_val:
            best_tour = cand_tour
            best_val = cand_val
            best_pack = cand_pack
            if verbose:
                print(f"improve: {best_val:.4f}")

    return best_tour, best_val, best_pack
