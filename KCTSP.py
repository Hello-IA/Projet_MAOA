import gurobipy as gp
from gurobipy import GRB
import numpy as np

from utils_io import TWDTSPLoader
from instance import KCTSPProblem
import pandas as pd

import random
import time
from typing import List, Tuple, Dict, Any


from utils_KCTSP import *

"""
def solve_KCTSP_LP(problem, D, time_limit=30, verbose=True):

    n = problem.n
    items = problem.m
    Wmax = problem.W
    KW = problem.kw

    model = gp.Model("KCTSP_LP")
    if not verbose:
        model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    # -------------------
    # Variables
    # -------------------
    x = model.addVars(n, n, lb=0, ub=1, name="x")   # arc i->j
    W = model.addVars(n, lb=0, ub=Wmax, name="W")   # poids cumulés après ville i
    z = model.addVars(n, n, lb=0, ub=Wmax, name="z")# poids transporté sur arc i->j

    y = {}
    for i, it in items.items():  # i = 0-based city
        for k in range(len(it)):
            y[i, k] = model.addVar(lb=0, ub=1, name=f"y_{i}_{k}")  # relaxation continue

    model.update()

    # -------------------
    # Contraintes TSP (degré)
    # -------------------
    for i in range(n):
        model.addConstr(gp.quicksum(x[i,j] for j in range(n) if j != i) == 1)
        model.addConstr(gp.quicksum(x[j,i] for j in range(n) if j != i) == 1)
        model.addConstr(W[i] <= Wmax)

    # -------------------
    # Contraintes sac-à-dos
    # -------------------
    model.addConstr(
        gp.quicksum(items[i][k][1] * y[i,k] for i in items for k in range(len(items[i]))) <= Wmax,
        name="capacity"
    )

    # -------------------
    # Contraintes McCormick pour z[i,j] = W[i] * x[i,j]
    # -------------------
    for i in range(n):
        for j in range(n):
            model.addConstr(z[i,j] <= W[i])
            model.addConstr(z[i,j] <= Wmax * x[i,j])
            model.addConstr(z[i,j] >= W[i] - Wmax * (1 - x[i,j]))
            model.addConstr(z[i,j] >= 0)

    # -------------------
    # Contraintes MTZ pour supprimer les sous-tours
    # -------------------
    u = model.addVars(n, lb=0, ub=n-1, name="u")
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i,j] <= n-1)

    # -------------------
    # Big-M pour W[j] = W[i] + poids objets à j si arc choisi
    # -------------------
    M = sum(items[i][k][1] for i in items for k in range(len(items[i])))
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(
                    W[j] >= W[i] + gp.quicksum(items.get(j,[])[k][1] * y[j,k] for k in range(len(items.get(j,[])))) - M*(1 - x[i,j])
                )

    # -------------------
    # Fonction objectif : profit - coût transport
    # -------------------
    obj_profit = gp.quicksum(items[i][k][0] * y[i,k] for i in items for k in range(len(items[i])))
    obj_transport = gp.quicksum(KW * D[i,j] * z[i,j] for i in range(n) for j in range(n) if i != j)

    model.setObjective(obj_profit - obj_transport, GRB.MAXIMIZE)

    model.optimize()

    return model, x, y, W
"""



# ------------------------------------------------------
# Utilitaires : format tour
# ------------------------------------------------------
def close_tour(order: List[int], start: int) -> List[int]:
    """order: permutation des villes (1-based) qui commence par start.
    Retourne un tour fermé: [..., start]
    """
    if order[0] != start:
        raise ValueError("order must start with start")
    if order[-1] == start:
        return order
    return order + [start]


def order_from_closed_tour(tour: List[int]) -> List[int]:
    """Enlève le dernier start (tour fermé) -> ordre permutation."""
    if len(tour) >= 2 and tour[0] == tour[-1]:
        return tour[:-1]
    return tour


# ------------------------------------------------------
# Évaluation (oracle)
# ------------------------------------------------------
def evaluate_tour(problem, order: List[int], start: int):
    """Retourne (fitness, packing_plan) pour un ordre (non fermé)."""
    tour = close_tour(order, start)
    packing_plan, fitness = greedy_kctsp_knapsack(problem, tour)  # doit renvoyer objectif net
    return fitness, packing_plan


# ------------------------------------------------------
# Initialisation population
# ------------------------------------------------------
def init_population(problem, pop_size: int, start: int, nn_seeds: int = 10) -> List[List[int]]:
    """
    - nn_seeds tours NN avec départs variés
    - le reste aléatoire
    """
    n = problem.n
    cities = list(range(1, n + 1))
    cities.remove(start)

    population = []

    # NN seeds depuis différents starts (si possible)
    # On garde start fixe comme "dépôt" (comme ton code), mais on varie le second point via permutation
    for _ in range(min(nn_seeds, pop_size)):
        nn_tour = tsp_nearest_neighbor(problem, start=start)  # fermé
        order = order_from_closed_tour(nn_tour)               # ordre
        population.append(order)

        # petite perturbation pour diversité
        if len(population) < pop_size and n > 5:
            pert = order[:]
            i = random.randint(1, n - 2)
            j = random.randint(1, n - 2)
            pert[i], pert[j] = pert[j], pert[i]
            population.append(pert)

        if len(population) >= pop_size:
            break

    # Compléter en aléatoire
    while len(population) < pop_size:
        perm = cities[:]
        random.shuffle(perm)
        order = [start] + perm
        population.append(order)

    return population[:pop_size]


# ------------------------------------------------------
# Sélection par tournoi
# ------------------------------------------------------
def tournament_select(population: List[List[int]], fitnesses: List[float], k: int = 3) -> List[int]:
    idxs = random.sample(range(len(population)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return population[best]


# ------------------------------------------------------
# Crossover OX (Order Crossover)
# ------------------------------------------------------
def order_crossover_OX(p1: List[int], p2: List[int], start: int) -> List[int]:
    """
    Parents p1,p2 : permutations 1-based commençant par start, non fermées.
    OX sur les positions 1..n-1 (on fige start à la position 0).
    """
    n = len(p1)
    assert p1[0] == start and p2[0] == start

    # Choisir segment sur [1, n-1)
    a = random.randint(1, n - 2)
    b = random.randint(a + 1, n - 1)

    child = [None] * n
    child[0] = start

    # Copier segment de p1
    child[a:b] = p1[a:b]
    used = set(child[a:b])
    used.add(start)

    # Remplir avec p2 dans l'ordre
    fill_positions = [i for i in range(1, n) if child[i] is None]
    fill_values = [c for c in p2[1:] if c not in used]

    for pos, val in zip(fill_positions, fill_values):
        child[pos] = val

    return child


# ------------------------------------------------------
# Mutations
# ------------------------------------------------------
def mutate_swap(order: List[int]) -> List[int]:
    n = len(order)
    if n <= 3:
        return order[:]
    new = order[:]
    i = random.randint(1, n - 1)
    j = random.randint(1, n - 1)
    new[i], new[j] = new[j], new[i]
    return new


def mutate_two_opt(order: List[int]) -> List[int]:
    n = len(order)
    if n <= 4:
        return order[:]
    new = order[:]
    i = random.randint(1, n - 3)
    j = random.randint(i + 1, n - 1)
    new[i:j] = reversed(new[i:j])
    return new


def mutate_relocate(order: List[int]) -> List[int]:
    n = len(order)
    if n <= 4:
        return order[:]
    new = order[:]
    i = random.randint(1, n - 1)
    city = new.pop(i)
    j = random.randint(1, n - 1)
    new.insert(j, city)
    return new


def mutate(order: List[int], p_swap=0.4, p_twoopt=0.4, p_reloc=0.2) -> List[int]:
    r = random.random()
    if r < p_swap:
        return mutate_swap(order)
    elif r < p_swap + p_twoopt:
        return mutate_two_opt(order)
    else:
        return mutate_relocate(order)


# ------------------------------------------------------
# GA principal
# ------------------------------------------------------
def solve_kctsp_ga(
    problem,
    start: int = 1,
    pop_size: int = 30,
    generations: int = 50,
    elite: int = 2,
    tournament_k: int = 3,
    p_crossover: float = 0.9,
    p_mutation: float = 0.3,
    time_limit: float = None,
    seed: int = 0,
    verbose: bool = True,
):
    random.seed(seed)

    # Init
    population = init_population(problem, pop_size=pop_size, start=start, nn_seeds=min(10, pop_size))
    fitnesses = []
    packings = []

    for ind in population:
        fit, pack = evaluate_tour(problem, ind, start)
        fitnesses.append(fit)
        packings.append(pack)

    best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
    best_order = population[best_idx][:]
    best_fit = fitnesses[best_idx]
    best_pack = packings[best_idx]

    t0 = time.time()

    for gen in range(generations):
        if time_limit is not None and (time.time() - t0) > time_limit:
            break

        # Trier population par fitness pour élite
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        new_population = [population[i][:] for i in ranked[:elite]]

        # Générer enfants
        while len(new_population) < pop_size:
            parent1 = tournament_select(population, fitnesses, k=tournament_k)
            parent2 = tournament_select(population, fitnesses, k=tournament_k)

            if random.random() < p_crossover:
                child = order_crossover_OX(parent1, parent2, start=start)
            else:
                child = parent1[:]

            if random.random() < p_mutation:
                child = mutate(child)

            new_population.append(child)

        # Évaluer nouvelle pop
        population = new_population
        fitnesses = []
        packings = []
        for ind in population:
            fit, pack = evaluate_tour(problem, ind, start)
            fitnesses.append(fit)
            packings.append(pack)

        # Mise à jour best
        gen_best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_order = population[gen_best_idx][:]
            best_pack = packings[gen_best_idx]

        if verbose:
            print(f"[Gen {gen+1:03d}] best={best_fit:.4f}  current_best={max(fitnesses):.4f}")

    best_tour = close_tour(best_order, start)
    return {
        "tour": best_tour,
        "packing_plan": best_pack,
        "objective": best_fit,
    }


def extract_tour_from_x(x, n, start=0):
    """
    Construire un tour entier à partir d'une solution fractionnaire du LP relaxé.
    
    Args:
        x : dict[(i,j)] -> valeur flottante [0,1] de la solution LP
        n : int, nombre de villes
        start : int, ville de départ (0-based)
    
    Returns:
        tour : list[int], tour entier (0-based)
    """
    tour = [start]
    visited = set(tour)

    current = start
    while len(tour) < n:
        # sélectionne les arcs fractionnaires vers les villes non visitées
        candidates = [(j, x[current, j].X) for j in range(n) if j not in visited]
        if not candidates:
            # cas rare : toutes les villes ont été visitées, on casse
            remaining = [j for j in range(n) if j not in visited]
            candidates = [(j, 0.0) for j in remaining]

        # choisir l'arc avec la valeur la plus élevée
        next_city = max(candidates, key=lambda t: t[1])[0]
        tour.append(next_city)
        visited.add(next_city)
        current = next_city

    # retour au dépôt
    tour.append(start)
    return tour


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
    



if __name__ == "__main__":
    db = pd.DataFrame(columns=[
        'problem_name', 'min_speed', 'max_speed', 'renting_ratio',
        'knapsack_data_type', 'dimension', 'num_items', 'max_weight', 'edge_weight_type'
    ])
    
    # Load a single file without populating database
    problem_tw = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n4460_bounded-strongly-corr_01.ttp")

    kw = 1.0/problem_tw.n   # coût par km et par kg (à choisir)
    problem_kc = problem_tw.as_kctsp(weight_cost_per_km=kw)
    """
    n = problem_kc.n
    D = np.zeros((n, n))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            D[i-1, j-1] = problem_kc.distance(i, j)



    model, x, y, W = solve_KCTSP_LP(problem_kc, D)

    tour = extract_tour_from_x(x, n)
    print(tour)

    packing_plan, obj_value = solve_kctsp_knapsack_exact(
        problem_kc,
        tour,
        time_limit=10
    )

    print("Valeur objective exacte :", obj_value)
    print("Objets sélectionnés :", packing_plan)
    """
    print("solve_kctsp_greedy", solve_kctsp_greedy(problem_kc))
    
    res = solve_kctsp_ga(problem_kc, start=1, pop_size=30, generations=50, elite=2, seed=0)
    print(res["objective"])
    print(len(res["packing_plan"]), "villes avec objets")
    print(res["tour"][:20], "...")
