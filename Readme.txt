KCTSP / TTP – Projet MAOA
=======================

Ce projet contient plusieurs algorithmes pour résoudre :
- le KCTSP (Knapsack-Constrained Traveling Salesman Problem)
- le TTP (Traveling Thief Problem)

Il inclut des heuristiques gloutonnes, un algorithme génétique et un algorithme mémétique.


----------------------------------
1. Lancer les algorithmes
----------------------------------

1) Heuristique gloutonne KCTSP (NN + glouton / PLNE)
-------------------------------------------------
Fichier :
    KCTSP_glouton.py

Commande :
    python KCTSP_glouton.py

Description :
    Construit un tour TSP avec Nearest Neighbor,
    puis sélectionne les objets avec un glouton ou un PLNE exact.


2) Algorithme génétique pour le KCTSP
-----------------------------------
Fichier :
    KCTSP.py

Commande :
    python KCTSP.py

Description :
    Utilise un algorithme génétique pour chercher de bons tours TSP.
    Chaque tour est évalué par un oracle de sac-à-dos (glouton ou PLNE).


3) Heuristique gloutonne pour le TTP
----------------------------------
Fichier :
    greedy_heuristic.py

Commande :
    python greedy_heuristic.py

Description :
    Construit un tour TSP (Lin-Kernighan),
    puis sélectionne les objets avec un glouton de sac-à-dos.


4) Algorithme mémétique MA2B pour le TTP
--------------------------------------
Fichier :
    MA2B.py

Commande :
    python MA2B.py

Description :
    Algorithme mémétique avec recherche locale TSP et sac-à-dos,
    basé sur l’article de Mei et al.


----------------------------------
2. Description des fichiers
----------------------------------

instance.py
    Définit les classes TTPProblem et KCTSPProblem (données, distances, coûts).

output.py
    Contient la classe TWDTSPSolution et la fonction display()
    pour afficher graphiquement un tour et les objets pris.

utils_KCTSP.py
    Fonctions utilitaires pour le KCTSP :
    Nearest Neighbor, glouton, PLNE, algorithme génétique.

KCTSP.py
    Implémentation de l’algorithme génétique pour le KCTSP.

KCTSP_glouton.py
    Implémentation du solveur KCTSP NN + glouton / PLNE.

greedy_heuristic.py
    Heuristique gloutonne pour le TTP.

KP_heuristic.py
    Heuristique d’insertion pour le sac-à-dos du TTP.

LK.py
    Implémentation de l’algorithme Lin-Kernighan pour le TSP.

MA2B.py
    Algorithme mémétique MA2B pour le TTP.


----------------------------------
3. Visualisation
----------------------------------

Les solutions peuvent être affichées sous forme d’image avec :
    solution.display(coords, items, path="plot/solution")

Cela génère un fichier PNG montrant le tour et la charge transportée.


----------------------------------
4. Données
----------------------------------

Les instances doivent être au format .ttp (CEC 2014).
Elles sont chargées avec :
    TWDTSPLoader.load_from_file("fichier.ttp")
