from output import TWDTSPSolution
from instance import TTPProblem
from utils_io import TWDTSPLoader
from LK import LKSolver
from KP_heuristic import insertion_heuristic
from greedy_heuristic import GreedyTTPSolver


import time
import numpy as np
import copy
from typing import List, Tuple, Dict

class MALTS:
    """
    MALTS: Memetic Algorithm with Late acceptance hill climbing and equal-Tsp Split
    for the Travelling Thief Problem
    """
    
    def __init__(self, 
                 problem: TTPProblem,
                 pop_size: int = 50,
                 num_generations: int = 100,
                 mutation_rate: float = 0.3,
                 lah_length: int = 100,
                 tsp_iterations: int = 500,
                 packing_iterations: int = 200,
                 seed: int = None):
        """
        Initialize MALTS algorithm
        
        Args:
            problem: TTP problem instance
            pop_size: Population size
            num_generations: Number of generations
            mutation_rate: Probability of mutation
            lah_length: Length of Late Acceptance Hill-climbing list
            tsp_iterations: Number of iterations for TSP local search
            packing_iterations: Number of iterations for packing plan improvement
            seed: Random seed for reproducibility
        """
        self.problem = problem
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.lah_length = lah_length
        self.tsp_iterations = tsp_iterations
        self.packing_iterations = packing_iterations
        
        if seed is not None:
            np.random.seed(seed)
        
        self.obj_func = problem.objective_function()
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    def solve(self) -> TWDTSPSolution:
        """
        Run the MALTS algorithm
        
        Returns:
            Best solution found
        """
        total_start = time.perf_counter()
        timing = {
            "initialization": [0.0],
            "evaluation": [0.0],
            "selection": [0.0],
            "crossover": [0.0],
            "mutation": [0.0],
            "local_search": [0.0],
            "generation_total": [0.0]
            }
        
        # Initialize population
        t0 = time.perf_counter()
        population = self._initialize_greedy_population()
        #population = self._initialize_population()
        timing["initialization"].append(time.perf_counter() - t0)
        
        for gen in range(self.num_generations):
            gen_start = time.perf_counter()

            # Evaluate population
            t0 = time.perf_counter()
            fitness_scores = [self.obj_func(sol) for sol in population]
            timing["evaluation"].append(time.perf_counter() - t0)
            
            # Update best solution
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[max_idx]
                self.best_solution = copy.deepcopy(population[max_idx])
            
            if gen % 2 == 0:
                print(f"Generation {gen}: Best fitness = {self.best_fitness:.2f}")
                print(f"Eval {timing['evaluation'][-1]:.2f}s | LS {timing['local_search'][-1]:.2f}s")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best solution
            new_population.append(copy.deepcopy(self.best_solution))
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                # Tournament selection
                t0 = time.perf_counter()
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                timing["selection"].append(time.perf_counter() - t0)
                
                # Crossover
                t0 = time.perf_counter()
                offspring = self._crossover(parent1, parent2)
                timing["crossover"].append(time.perf_counter() - t0)
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    t0 = time.perf_counter()
                    offspring = self._mutate(offspring)
                    timing["mutation"].append(time.perf_counter() - t0)
                
                # Local search with Late Acceptance Hill-climbing
                t0 = time.perf_counter()
                offspring = self._late_acceptance_hill_climbing(offspring)
                timing["local_search"].append(time.perf_counter() - t0)
                
                new_population.append(offspring)
            
            population = new_population

            timing["generation_total"].append(time.perf_counter() - gen_start)
        
        total_time = time.perf_counter() - total_start

        print("\n==== TIMING SUMMARY ====")
        print(f"Total runtime:        {total_time:.2f}s")
        print(f"Initialization:      {timing['initialization'][0]:.2f}s")
        print(f"Evaluation:          total={sum(timing['evaluation']):.2f}s, avg={sum(timing['evaluation'])/self.num_generations}s")
        print(f"Selection:           total={sum(timing['selection']):.2f}s, avg={sum(timing['selection'])/self.num_generations}s")
        print(f"Crossover:           total={sum(timing['crossover']):.2f}s, avg={sum(timing['crossover'])/self.num_generations}s")
        print(f"Mutation:            total={sum(timing['mutation']):.2f}s, avg={sum(timing['mutation'])/self.num_generations}s")
        print(f"Local search:        total={sum(timing['local_search']):.2f}s, avg={sum(timing['local_search'])/self.num_generations}s")
        print(f"Generation overhead: total={sum(timing['generation_total']):.2f}s, avg={sum(timing['generation_total'])/self.num_generations}s")
        print("========================\n")
        
        return self.best_solution
    
    def _initialize_greedy_population(self) -> List[TWDTSPSolution]:
        population = []
        greedySolver = GreedyTTPSolver(self.problem)
        solution = greedySolver.tsp_first(group_size=self.problem.n // 5)
        tour = solution.tour
        print(tour)
        for i in range(self.pop_size - 1):
            solution = greedySolver.tsp_first(group_size=self.problem.n // 5, seed=i, tour=tour)
            population.append(solution)
        return population

    
    def _initialize_population(self) -> List[TWDTSPSolution]:
        """
        Initialize population with random solutions
        """
        population = []

        
        for i in range(self.pop_size):
            # Create random tour
            tour = self._create_random_tour()
            if i == 0:
                lkSolver = LKSolver(self.problem.coords)
                tour = lkSolver.lk(max_iter=1)
                tour.append(tour[0])
            
            # Create packing plan using greedy heuristic
            #packing_plan = self._greedy_packing(tour)
            packing_plan = insertion_heuristic(self.problem, tour)
            
            # Calculate totals
            total_profit, total_weight = self._calculate_totals(packing_plan)
            
            solution = TWDTSPSolution(
                tour=tour,
                packing_plan=packing_plan,
                total_profit=total_profit,
                total_weight=total_weight,
                max_weight=self.problem.W
            )
            
            population.append(solution)
        
        return population
    
    def _create_random_tour(self) -> List[int]:
        """
        Create a random Hamiltonian cycle starting and ending at city 1
        """
        cities = list(range(1, self.problem.n + 1))
        np.random.shuffle(cities)
        
        # Ensure tour starts and ends at city 1
        if cities[0] != 1:
            idx = cities.index(1)
            cities[0], cities[idx] = cities[idx], cities[0]
        
        tour = cities + [cities[0]]
        return tour

    def _greedy_packing(self, tour: List[int]) -> np.ndarray:
        """
        Create packing plan using tour-aware greedy heuristic
        """
        max_items = self.problem.p.shape[1]
        packing_plan = np.zeros((self.problem.n, max_items), dtype=int)
        current_weight = 0.0
    
        remaining_distance = {}
        for i in range(len(tour) - 1, 0, -1):
            city_from = tour[i]
            city_to = tour[i-1]
            dist = self.problem.distance(city_from, city_to)
            remaining_distance[tour[i-1]] = remaining_distance.get(tour[i], 0) + dist
    
        # Create list with tour-aware scoring
        items_list = []
        for city_idx in range(self.problem.n):
            rem_dist = remaining_distance.get(city_idx + 1, 0)
        
            for item_idx in range(max_items):
                profit = self.problem.p[city_idx, item_idx]
                weight = self.problem.w[city_idx, item_idx]
            
                if weight > 0 and rem_dist > 0:
                    # Penalize items picked up early
                    score = profit / (weight * rem_dist + 1e-6)
                    items_list.append((score, city_idx, item_idx, profit, weight))
    
        items_list.sort(reverse=True)
    
        for score, city_idx, item_idx, profit, weight in items_list:
            if current_weight + weight <= self.problem.W:
                packing_plan[city_idx, item_idx] = 1
                current_weight += weight
    
        return packing_plan
    
    def _calculate_totals(self, packing_plan: np.ndarray) -> Tuple[float, float]:
        """
        Calculate total profit and weight from packing plan
        """
        total_profit = np.sum(packing_plan * self.problem.p)
        total_weight = np.sum(packing_plan * self.problem.w)
        return total_profit, total_weight
    
    def _tournament_selection(self, population: List[TWDTSPSolution], 
                             fitness_scores: List[float], 
                             tournament_size: int = 3) -> TWDTSPSolution:
        """
        Select individual using tournament selection
        """
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return copy.deepcopy(population[best_idx])
    
    def _crossover(self, parent1: TWDTSPSolution, 
                   parent2: TWDTSPSolution) -> TWDTSPSolution:
        """
        Order crossover (OX) for tour + uniform crossover for packing
        """
        # Order crossover for tour
        tour1 = parent1.tour[:-1]  # Remove duplicate last city
        tour2 = parent2.tour[:-1]
        
        size = len(tour1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        # Create offspring tour
        offspring_tour = [-1] * size
        offspring_tour[start:end] = tour1[start:end]
        
        # Fill remaining positions from parent2
        fill_pos = end
        for city in tour2[end:] + tour2[:end]:
            if city not in offspring_tour:
                if fill_pos >= size:
                    fill_pos = 0
                offspring_tour[fill_pos] = city
                fill_pos += 1
        
        # Add closing city
        offspring_tour.append(offspring_tour[0])
        
        # Uniform crossover for packing plan
        mask = np.random.random(parent1.packing_plan.shape) < 0.5
        offspring_packing = np.where(mask, parent1.packing_plan, parent2.packing_plan)
        
        # Ensure weight constraint
        total_weight = np.sum(offspring_packing * self.problem.w)
        
        # If over weight, remove items randomly
        while total_weight > self.problem.W:
            packed_items = np.argwhere(offspring_packing == 1)
            if len(packed_items) == 0:
                break
            
            idx = np.random.randint(len(packed_items))
            city_idx, item_idx = packed_items[idx]
            offspring_packing[city_idx, item_idx] = 0
            total_weight -= self.problem.w[city_idx, item_idx]
        
        total_profit, total_weight = self._calculate_totals(offspring_packing)
        
        return TWDTSPSolution(
            tour=offspring_tour,
            packing_plan=offspring_packing,
            total_profit=total_profit,
            total_weight=total_weight,
            max_weight=self.problem.W
        )
    
    def _mutate(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        Apply mutation operators
        """
        solution = copy.deepcopy(solution)
        
        # 50% chance to mutate tour, 50% chance to mutate packing
        if np.random.random() < 0.5:
            # 2-opt mutation for tour
            solution = self._two_opt_mutation(solution)
        else:
            # Bit-flip mutation for packing
            solution = self._packing_mutation(solution)
        
        return solution
    
    def _two_opt_mutation(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        Apply 2-opt mutation to tour
        """
        tour = solution.tour[:-1]  # Remove duplicate last city
        size = len(tour)
        
        i, j = sorted(np.random.choice(size, 2, replace=False))
        
        # Reverse segment
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        new_tour.append(new_tour[0])
        
        total_profit, total_weight = self._calculate_totals(solution.packing_plan)
        
        return TWDTSPSolution(
            tour=new_tour,
            packing_plan=solution.packing_plan.copy(),
            total_profit=total_profit,
            total_weight=total_weight,
            max_weight=self.problem.W
        )
    
    def _packing_mutation(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        Flip random bits in packing plan
        """
        packing = solution.packing_plan.copy()
        max_items = packing.shape[1]
        
        # Flip 1-3 random items
        num_flips = np.random.randint(1, 4)
        
        for _ in range(num_flips):
            city_idx = np.random.randint(self.problem.n)
            
            # Find valid items at this city (items with non-zero weight)
            valid_items = np.where(self.problem.w[city_idx, :] > 0)[0]
            
            if len(valid_items) > 0:
                item_idx = np.random.choice(valid_items)
                packing[city_idx, item_idx] = 1 - packing[city_idx, item_idx]
        
        # Repair if over weight
        total_weight = np.sum(packing * self.problem.w)
        
        while total_weight > self.problem.W:
            packed_items = np.argwhere(packing == 1)
            if len(packed_items) == 0:
                break
            
            idx = np.random.randint(len(packed_items))
            city_idx, item_idx = packed_items[idx]
            packing[city_idx, item_idx] = 0
            total_weight -= self.problem.w[city_idx, item_idx]
        
        total_profit, total_weight = self._calculate_totals(packing)
        
        return TWDTSPSolution(
            tour=solution.tour.copy(),
            packing_plan=packing,
            total_profit=total_profit,
            total_weight=total_weight,
            max_weight=self.problem.W
        )
    
    def _late_acceptance_hill_climbing(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        Apply Late Acceptance Hill-climbing local search
        """
        current = copy.deepcopy(solution)
        current_fitness = self.obj_func(current)
        
        # Initialize fitness history
        fitness_history = [current_fitness] * self.lah_length
        
        for iteration in range(self.tsp_iterations + self.packing_iterations):
            if iteration < self.tsp_iterations:
                # TSP improvement: 2-opt
                neighbor = self._two_opt_neighbor(current)
            else:
                # Packing improvement: bit-flip
                neighbor = self._packing_neighbor(current)
            
            neighbor_fitness = self.obj_func(neighbor)
            
            # Late acceptance criterion
            v = iteration % self.lah_length
            
            if neighbor_fitness >= fitness_history[v]:
                current = neighbor
                current_fitness = neighbor_fitness
            
            fitness_history[v] = current_fitness
        
        return current
    
    def _two_opt_neighbor(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        Generate neighbor using 2-opt move
        """
        tour = solution.tour[:-1]
        size = len(tour)
        
        i, j = sorted(np.random.choice(size, 2, replace=False))
        
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        new_tour.append(new_tour[0])
        
        total_profit, total_weight = self._calculate_totals(solution.packing_plan)
        
        return TWDTSPSolution(
            tour=new_tour,
            packing_plan=solution.packing_plan.copy(),
            total_profit=total_profit,
            total_weight=total_weight,
            max_weight=self.problem.W
        )
    
    def _packing_neighbor(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        Generate neighbor by flipping one item
        """
        packing = solution.packing_plan.copy()
        max_items = packing.shape[1]
        
        # Try to flip a random item
        city_idx = np.random.randint(self.problem.n)
        
        # Find valid items at this city
        valid_items = np.where(self.problem.w[city_idx, :] > 0)[0]
        
        if len(valid_items) > 0:
            item_idx = np.random.choice(valid_items)
            
            current_val = packing[city_idx, item_idx]
            weight_change = self.problem.w[city_idx, item_idx]
            current_weight = np.sum(packing * self.problem.w)
            
            # Check if flip is feasible
            if current_val == 1 or current_weight + weight_change <= self.problem.W:
                packing[city_idx, item_idx] = 1 - current_val
        
        total_profit, total_weight = self._calculate_totals(packing)
        
        return TWDTSPSolution(
            tour=solution.tour.copy(),
            packing_plan=packing,
            total_profit=total_profit,
            total_weight=total_weight,
            max_weight=self.problem.W
        )


# Example usage
if __name__ == "__main__":

    #problem = TWDTSPLoader.load_from_file("./cities280/a280_n279_bounded-strongly-corr_01.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities280/a280_n1395_uncorr-similar-weights_05.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities280/a280_n2790_uncorr_10.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n4460_bounded-strongly-corr_01.ttp.txt", asTTP=True)
    problem = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n22300_uncorr-similar-weights_05.ttp.txt", asTTP=True)


    # Run MALTS
    malts = MALTS(
        problem=problem,
        pop_size=30,
        num_generations=50,
        mutation_rate=0.3,
        lah_length=5,
        tsp_iterations=20,
        packing_iterations=20,
        seed=42
    )
    
    best_solution = malts.solve()
    
    print("\n" + "="*50)
    print("MALTS Algorithm - Best Solution Found")
    print("="*50)
    print(best_solution)
    print(f"\nObjective value: {malts.best_fitness:.2f}")
    print(f"Items collected (city, index, profit, weight): {best_solution.get_collected_items(problem.m)}")