from utils_io import TWDTSPLoader
from instance import KCTSPProblem
import pandas as pd
import numpy as np

db = pd.DataFrame(columns=[
        'problem_name', 'min_speed', 'max_speed', 'renting_ratio',
        'knapsack_data_type', 'dimension', 'num_items', 'max_weight', 'edge_weight_type'
    ])
    
# Load a single file without populating database
problem_tw = TWDTSPLoader.load_from_file("./data/a280_n279_bounded-strongly-corr_01.ttp")

kw = 0.01  # coût par km et par kg (à choisir)
problem_kc = problem_tw.as_kctsp(weight_cost_per_km=kw)

n = problem_kc.n
D = np.zeros((n, n))

for i in range(1, n + 1):
    for j in range(1, n + 1):
        D[i-1, j-1] = problem_kc.distance(i, j)

print(D.shape)


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
    Glouton pour le KCTSP en prenant en compte le coût du transport dans la fonction objectif.
    
    Args:
        problem : KCTSPProblem
        tour : list[int], ordre des villes à visiter (1-based)
        
    Returns:
        packing_plan : dict[int, list[int]], objets collectés par ville
        total_profit : float, profit total réel
        total_transport_cost : float, coût cumulatif du transport
    """
    n = len(tour)
    packing_plan = {}
    carried_weight = 0.0
    total_profit = 0.0
    total_transport_cost = 0.0

    # On parcourt le tour
    for idx, city in enumerate(tour):
        items = problem.city_items(city)
        if len(items) == 0:
            continue

        profit = items[:, 0]
        weight = items[:, 1]

        # On calcule pour chaque objet l'impact sur la fonction objectif
        # c'est-à-dire son profit réel moins le coût de transport additionnel si on le prend
        # Le coût de transport supplémentaire = poids de l'objet * KW * distance restante
        dist_restante = 0.0
        # somme des distances jusqu'à la fin du tour
        for k in range(idx, n-1):
            dist_restante += problem.distance(tour[k], tour[k+1])

        # profit net = profit - KW * weight * dist_restante
        profit_net = profit - problem.kw * weight * dist_restante

        # Trier les objets par profit net décroissant
        sorted_idx = np.argsort(-profit_net)

        selected = []
        for i in sorted_idx:
            if carried_weight + weight[i] <= problem.W and profit_net[i] > 0:
                # on ne prend que si cela améliore la fonction objectif (profit net positif)
                selected.append(i)
                carried_weight += weight[i]
                total_profit += profit[i]  # profit réel
        if selected:
            packing_plan[city] = selected

        # Mettre à jour le coût cumulatif pour le transport jusqu'à la prochaine ville
        if idx < n - 1:
            next_city = tour[idx + 1]
            total_transport_cost += problem.transport_cost(city, next_city, carried_weight)
    
    objective_value = total_profit - total_transport_cost

    return packing_plan, objective_value, total_transport_cost



# Supposons que problem_kc est un KCTSPProblem
tour = NearestNeighbor(D, start=1)
packing_plan, profit, transport_cost = KnapsackGloutonKCTSP(problem_kc, tour)

print("Tour :", tour)
print("Packing plan :", packing_plan)
print("Profit total :", profit)
print("Coût total transport :", transport_cost)


        
