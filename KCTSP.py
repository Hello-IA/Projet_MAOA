from utils_io import TWDTSPLoader
from instance import KCTSPProblem
from output import TWDTSPSolution
import pandas as pd

import random
import time
from typing import List


from utils_KCTSP import *


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
def init_population(problem, pop_size: int, start: int, n_perturb: int = 20, nn_seeds: int = 10) -> List[List[int]]:
    """
    - nn_seeds tours NN avec départs variés
    - le reste aléatoire
    """
    n = problem.n
    cities = list(range(1, n + 1))
    cities.remove(start)

    population = []


    nn_tour = tsp_nearest_neighbor(problem, start=start)  # ferme
    nn_order = order_from_closed_tour(nn_tour)            # non ferme
    population.append(nn_order)

    # Generer des variantes rapides (pas de NN)
    while len(population) < min(pop_size, 1 + n_perturb):
        child = nn_order[:]

        r = random.random()
        if r < 0.34:
            child = mutate_swap(child)
        elif r < 0.67:
            child = mutate_two_opt(child)
        else:
            child = mutate_relocate(child)

        population.append(child)

    # Completer en aleatoire
    while len(population) < pop_size:
        perm = cities[:]
        random.shuffle(perm)
        order = [start] + perm
        population.append(order)


    return population[:pop_size]



# Selection par tournoi
def tournament_select(population: List[List[int]], fitnesses: List[float], k: int = 3) -> List[int]:
    idxs = random.sample(range(len(population)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return population[best]



# Crossover OX (Order Crossover)
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



# Mutations
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



# GA principal
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
    verbose: bool = True):
    random.seed(seed)

    # Init
    population = init_population(problem, pop_size=pop_size, start=start, n_perturb = 20, nn_seeds=min(10, pop_size))
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
    

    for gen in range(generations):
        start_time = time.time()
        if time_limit is not None and (time.time() - t0) > time_limit:
            break

        # Trier population par fitness pour élite
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        new_population = [population[i][:] for i in ranked[:elite]]

        # Generer enfants
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

        # Evaluer nouvelle pop
        population = new_population
        fitnesses = []
        packings = []
        for ind in population:
            fit, pack = evaluate_tour(problem, ind, start)
            fitnesses.append(fit)
            packings.append(pack)

        # Mise a jour best
        gen_best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_order = population[gen_best_idx][:]
            best_pack = packings[gen_best_idx]

        if verbose:
            print(f"[Gen {gen+1:03d}] best={best_fit:.4f}  current_best={max(fitnesses):.4f}")
        end_time = time.time()


    best_tour = close_tour(best_order, start)
    return {
        "tour": best_tour,
        "packing_plan": best_pack,
        "objective": best_fit,
    }



def solve_kctsp_greedy(problem, start_city=1):
    tour = tsp_nearest_neighbor(problem, start=start_city)

    packing_plan, profit = greedy_kctsp_knapsack(problem, tour)

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
    problem_tw = TWDTSPLoader.load_from_file("./cities280/a280_n279_bounded-strongly-corr_01.ttp")

    kw = 1.0/problem_tw.n   # coot par km et par kg (a choisir)
    problem_kc = problem_tw.as_kctsp(weight_cost_per_km=kw)
    
    #print("solve_kctsp_greedy", solve_kctsp_greedy(problem_kc))
    
    
    res = solve_kctsp_ga(problem_kc, start=1, pop_size=30, generations=200, elite=2, seed=0)


    print(res["objective"])
    print(len(res["packing_plan"]), "villes avec objets")
    print(res["tour"][:20], "...")
    