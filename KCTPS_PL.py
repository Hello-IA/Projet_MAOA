import pandas as pd
from utils_io import TWDTSPLoader
from utils_KCTSP import *


"""
Se code ne marche pas cette une movaise idée de faire un relaxations du PLNE et de ensuite construire un tour a partire de la relaxation. 
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



def extract_tour_from_x(x, n, start=0):
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


if __name__ == "naim":
    
    db = pd.DataFrame(columns=[
        'problem_name', 'min_speed', 'max_speed', 'renting_ratio',
        'knapsack_data_type', 'dimension', 'num_items', 'max_weight', 'edge_weight_type'
    ])
    
    # Load a single file without populating database
    problem_tw = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n4460_bounded-strongly-corr_01.ttp")

    kw = 1.0/problem_tw.n   # coût par km et par kg (à choisir)
    problem_kc = problem_tw.as_kctsp(weight_cost_per_km=kw)
    
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