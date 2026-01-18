from utils_io import TWDTSPLoader
from instance import KCTSPProblem
import pandas as pd
import numpy as np
import time
from KCTSP import solve_kctsp_knapsack_exact


def NearestNeighbor(D, start=0):
    """
    Algorithme du plus proche voisin pour TSP.
    
    Args:
        D : np.ndarray, matrice de distances (n x n)
        start : int, ville de départ (0-based)
    
    Returns:
        tour : list d'entiers, ordre des villes visitées
    """
    n = D.shape[0]
    tour = [start]            # commence avec la ville de départ
    visited = set(tour)       # ensemble pour vérifier rapidement les villes déjà visitées
    current = start

    while len(tour) < n:
        # copie des distances depuis la ville courante
        dist = D[current].copy()

        # mettre les villes déjà visitées à l'infini pour les exclure
        dist[list(visited)] = np.inf

        # prendre l'indice de la ville la plus proche non visitée
        next_city = np.argmin(dist)

        # ajouter la ville au tour
        tour.append(next_city)
        visited.add(next_city)

        # mettre à jour la ville courante
        current = next_city

    return tour



def KnapsackGloutonKCTSP(problem: KCTSPProblem, tour):
    """
    Glouton EXACT pour le KCTSP le long d'un tour donné.
    Calcule le profit net exact en prenant en compte le poids cumulatif transporté.

    Args:
        problem : KCTSPProblem
        tour : list[int], ordre des villes à visiter (0-based)

    Returns:
        packing_plan : dict[int, list[int]], objets collectés par ville
        objective_value : float, valeur exacte de la fonction objectif
        total_transport_cost : float
    """
    n = len(tour)
    packing_plan = {}
    carried_weight = 0.0
    total_transport_cost = 0.0
    total_profit = 0.0

    # Poids cumulatif transporté après chaque ville
    W_cum = np.zeros(n)

    # On parcourt le tour
    for idx, city in enumerate(tour):
        items = problem.city_items(city)
        if len(items) == 0:
            continue

        profit = items[:, 0]
        weight = items[:, 1]

        selected = []

        # Tester chaque objet séparément
        for i in range(len(items)):
            w_obj = weight[i]
            p_obj = profit[i]

            # Vérifier la contrainte de sac-à-dos
            if carried_weight + w_obj > problem.W:
                continue

            # Calculer le coût de transport additionnel si on prend cet objet
            transport_add = 0.0
            for k in range(idx, n-1):
                transport_add += problem.distance(tour[k], tour[k+1]) * w_obj * problem.kw

            # Profit net si on prend cet objet
            net_profit = p_obj - transport_add

            if net_profit > 0:
                selected.append(i)
                carried_weight += w_obj
                total_profit += p_obj

                # Mettre à jour le poids cumulatif
                for k in range(idx, n):
                    W_cum[k] += w_obj

        if selected:
            packing_plan[city] = selected

        # Mettre à jour le coût cumulatif pour le transport jusqu'à la prochaine ville
        if idx < n - 1:
            next_city = tour[idx + 1]
            total_transport_cost += problem.kw * W_cum[idx] * problem.distance(city, next_city)

    objective_value = total_profit - total_transport_cost
    return packing_plan, objective_value, total_transport_cost








if __name__ == "__main__":
    
    db = pd.DataFrame(columns=[
        'problem_name', 'min_speed', 'max_speed', 'renting_ratio',
        'knapsack_data_type', 'dimension', 'num_items', 'max_weight', 'edge_weight_type'
    ])
    
    # Load a single file without populating database
    problem_tw = TWDTSPLoader.load_from_file("./cities33810/pla33810_n33809_bounded-strongly-corr_01.ttp")

    kw = 0.01  # coût par km et par kg (à choisir)
    problem_kc = problem_tw.as_kctsp(weight_cost_per_km=kw)
    start = time.time()

    n = problem_kc.n
    D = np.zeros((n, n))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            D[i-1, j-1] = problem_kc.distance(i, j)
    end = time.time()
    print("Time D :", end-start)



    # Supposons que problem_kc est un KCTSPProblem
    start = time.time()
    
    tour = NearestNeighbor(D, start=1)
    end = time.time()
    print("Time NearestNeighbor :", end-start)
    
    start = time.time()
    packing_plan, profit = solve_kctsp_knapsack_exact(
        problem_kc,
        tour,
        time_limit=10
    )
    end = time.time()
    print("Time solve_kctsp_knapsack_exact :", end-start)
    #packing_plan, profit, transport_cost = KnapsackGloutonKCTSP(problem_kc, tour)

    #print("Tour :", tour)
    print("Packing plan :", packing_plan)
    print("Profit total :", profit)
    #print("Coût total transport :", transport_cost)


        
