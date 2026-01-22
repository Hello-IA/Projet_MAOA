import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any

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
    tour : list[int] 1-based et ferme (dernier == premier)
    retourne: dict city(1-based) -> distance restante depuis cette ville (jusqu'au retour depot)
    """
    # recuper la liste des cordonner de noeud du graphe
    coords = problem.coords  # shape (n,2)

    # convertir tour (1-based) en indices 0-based
    t = np.asarray(tour, dtype=np.int32) - 1  # shape (L,)
    # aretes consecutives (je suprime le dernier element de all_less_end et le premeier de all_less_first)
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

    # remaining pour chaque position i (sauf derniere, qui est le retour au start)
    # remaining[i] = total - sum(dist[:i])
    #calcule la liste de la some cumuler
    #a = np.array([[1,2,3], [4,5,6]])
    #np.cumsum(a) -> array([ 1,  3,  6, 10, 15, 21])
    prefix = np.concatenate(([0.0], np.cumsum(dist)))  
    #on retire au totale la somme cumuler de 0 jusqua i pour remaining_vals[i] sa lese la somme de i a n-1
    remaining_vals = total - prefix[:-1]               

    # map ville -> remaining (attention: une ville apparait une seule fois dans un tour TSP)
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
        #on aplique un masque sur pour navoir que se qui on un profi_net superieur a 0
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
    #on range en liste d'objet a recuperer 
    ratios = np.concatenate(ratios_all)
    nets = np.concatenate(net_all)
    ws = np.concatenate(w_all)
    idxs = np.concatenate(idx_all)
    cities = np.concatenate(cities_all)
    # on tris dans un ordre decroisent des ratios
    order = np.argsort(-ratios)  # decroissant
    # on aplique l'agorithme glouton
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




def solve_kctsp_knapsack_exact(problem, tour, time_limit=30, verbose=True):
    """
    Resout exactement le KCTSP pour un tour fixe via PLNE (Gurobi).
    """
    remaining_dist = compute_remaining_distances(problem, tour)

    m = gp.Model("KCTSP_Knapsack_FixedTour")
    #choisi si on afiche ou non les log de Gurobis
    if not verbose:
        m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    #variable binaire y[(city, j)] 1 si on prent l'objet j dans la vile city et 0 sinon
    y = {}   # variable par objet
    obj_terms = []

    # creation des variables
    for city in tour:
        items = problem.city_items(city)
        if len(items) == 0:
            continue
        #si la vile posaide des objet
        for j, (profit, weight) in enumerate(items):
            var_name = f"y_{city}_{j}"
            y[(city, j)] = m.addVar(vtype=GRB.BINARY, name=var_name)

            net_profit = profit - problem.kw * weight * remaining_dist[city]
            obj_terms.append(net_profit * y[(city, j)])

    # fonction objectif
    m.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)

    # contrainte de capacite du sac a dos
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







