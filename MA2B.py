from output import TWDTSPSolution
from instance import TTPProblem
from utils_io import TWDTSPLoader
from KP_heuristic import insertion_heuristic

import time
import numpy as np
import copy
from typing import List, Tuple, Dict


class MA2B:
    """
    MA2B: Memetic Algorithm with Two-stage local search (Based on MATLS)
    Implements the algorithm from "Improving Efficiency of Heuristics for the 
    Large Scale Traveling Thief Problem" by Mei et al.
    
    Key features:
    - Two-stage local search: TSP stage + KP stage
    - Fitness approximation for TSP stage (minimize tour length)
    - Insertion heuristic with three approximations for KP stage
    - Ordered crossover for tours
    """
    
    def __init__(self, 
                 problem,  # TTPProblem instance
                 pop_size: int = 30,
                 num_generations: int = 100,
                 tsp_iterations: int = 50,
                 kp_iterations: int = 10,
                 seed: int = None):
        """
        Initialize MA2B algorithm
        
        Args:
            problem: TTP problem instance
            pop_size: Population size
            num_generations: Number of generations
            tsp_iterations: Number of 2-opt iterations for TSP stage
            kp_iterations: Number of flip iterations for KP stage
            seed: Random seed for reproducibility
        """
        self.problem = problem
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.tsp_iterations = tsp_iterations
        self.kp_iterations = kp_iterations
        
        if seed is not None:
            np.random.seed(seed)
        
        self.obj_func = problem.objective_function()
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    def solve(self) -> TWDTSPSolution:
        """
        Run the MA2B algorithm
        """
        total_start = time.perf_counter()

        initialization_time = [0.0]
        evaluation_time = [0.0]
        selection_time = [0.0]
        crossover_time = [0.0]
        local_search_time = [0.0]
        generation_time = [0.0]
        
        # Initialize population
        print("Initializing population...")
        t0 = time.perf_counter()
        population = self._initialize_population()
        initialization_time.append(time.perf_counter() - t0)
        
        for gen in range(self.num_generations):
            generation_t0 = 0
            # Evaluate population
            t0 = time.perf_counter()
            fitness_scores = [self.obj_func(sol) for sol in population]
            evaluation_time.append(time.perf_counter() - t0)
            
            # Update best solution
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[max_idx]
                self.best_solution = copy.deepcopy(population[max_idx])
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {self.best_fitness:.2f}")
            
            # Generate offspring
            # Select two random parents
            t0 = time.perf_counter()
            parent1_idx = np.random.randint(len(population))
            parent2_idx = np.random.randint(len(population))
            selection_time.append(time.perf_counter() - t0)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover (ordered crossover for tour)
            t0 = time.perf_counter()
            offspring = self._ordered_crossover(parent1, parent2)
            crossover_time.append(time.perf_counter() - t0)
            
            # Two-stage local search
            t0 = time.perf_counter()
            offspring = self._two_stage_local_search(offspring)
            local_search_time.append(time.perf_counter() - t0)
            
            # Replace worst individual
            worst_idx = np.argmin(fitness_scores)
            population[worst_idx] = offspring

            generation_time.append(time.perf_counter() - generation_t0)
        
        total_time = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_time:.2f}s")
        
        print(f"Initialization:      {initialization_time[0]:.2f}s")
        print(f"Evaluation:          total={sum(evaluation_time):.2f}s, avg={sum(evaluation_time)/self.num_generations}s")
        print(f"Selection:           total={sum(selection_time):.2f}s, avg={sum(selection_time)/self.num_generations}s")
        print(f"Crossover:           total={sum(crossover_time):.2f}s, avg={sum(crossover_time)/self.num_generations}s")
        print(f"Local search:        total={sum(local_search_time):.2f}s, avg={sum(local_search_time)/self.num_generations}s")
        print(f"Generation overhead: total={sum(generation_time):.2f}s, avg={sum(generation_time)/self.num_generations}s")
        
        return self.best_solution
    
    def _initialize_population(self) -> List[TWDTSPSolution]:
        """
        Initialize population with nearest neighbor heuristic + insertion heuristic
        """
        population = []
        
        for i in range(self.pop_size):
            if i < self.pop_size // 2:
                # Use nearest neighbor heuristic for half
                tour = self._nearest_neighbor_tour()
            else:
                # Random tours for other half
                tour = self._create_random_tour()
            
            # Apply insertion heuristic for packing
            packing_plan = insertion_heuristic(self.problem, tour)
            
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
    
    def _nearest_neighbor_tour(self) -> List[int]:
        """
        Create tour using nearest neighbor heuristic
        """
        n = self.problem.n
        unvisited = set(range(1, n + 1))
        tour = [1]  # Start at city 1
        unvisited.remove(1)
        
        current = 1
        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda city: self.problem.distance(current, city))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        tour.append(1)  # Return to start
        return tour
    
    def _create_random_tour(self) -> List[int]:
        """
        Create a random Hamiltonian cycle starting and ending at city 1
        """
        cities = list(range(1, self.problem.n + 1))
        np.random.shuffle(cities)
        
        if cities[0] != 1:
            idx = cities.index(1)
            cities[0], cities[idx] = cities[idx], cities[0]
        
        tour = cities + [cities[0]]
        return tour
    
    def _calculate_totals(self, packing_plan: np.ndarray) -> Tuple[float, float]:
        """
        Calculate total profit and weight from packing plan (just for solution display)
        """
        total_profit = np.sum(packing_plan * self.problem.p)
        total_weight = np.sum(packing_plan * self.problem.w)
        return total_profit, total_weight
    
    def _ordered_crossover(self, parent1: TWDTSPSolution, 
                          parent2: TWDTSPSolution) -> TWDTSPSolution:
        """
        Ordered crossover (OX) for tour
        """
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
        
        # Initialize with empty packing (will be filled by insertion heuristic)
        max_items = self.problem.p.shape[1]
        packing_plan = np.zeros((self.problem.n, max_items), dtype=int)
        
        return TWDTSPSolution(
            tour=offspring_tour,
            packing_plan=packing_plan,
            total_profit=0.0,
            total_weight=0.0,
            max_weight=self.problem.W
        )
    
    def _two_stage_local_search(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        Two-stage local search
        """
        current = copy.deepcopy(solution)
        
        # Stage 1: TSP search with approximated fitness (tour length)
        current = self._tsp_local_search(current)
        
        # Stage 2: KP search with insertion heuristic + local search
        current.packing_plan = insertion_heuristic(self.problem, current.tour)
        current.total_profit, current.total_weight = self._calculate_totals(current.packing_plan)
        
        # Refine packing with bit-flip local search
        current = self._kp_local_search(current)
        
        return current
    
    def _tsp_local_search(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        TSP local search using 2-opt with tour length minimization
        """
        current = copy.deepcopy(solution)
        improved = True
        iterations = 0
        
        while improved and iterations < self.tsp_iterations:
            improved = False
            iterations += 1
            
            tour = current.tour[:-1]
            n = len(tour)
            
            # Try 2-opt moves
            for _ in range(min(n, 20)):  # Limit attempts per iteration
                i = np.random.randint(0, n - 1)
                j = np.random.randint(i + 1, n)
                
                # Calculate current edge lengths
                current_length = (
                    self.problem.distance(tour[i], tour[i + 1]) +
                    self.problem.distance(tour[j], tour[(j + 1) % n])
                )
                
                # Calculate new edge lengths after reversal
                new_length = (
                    self.problem.distance(tour[i], tour[j]) +
                    self.problem.distance(tour[i + 1], tour[(j + 1) % n])
                )
                
                # Accept if improvement
                if new_length < current_length:
                    tour = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
                    current.tour = tour + [tour[0]]
                    improved = True
                    break
        
        return current
    
    def _kp_local_search(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        KP local search using bit-flip operator
        """
        current = copy.deepcopy(solution)
        current_fitness = self.obj_func(current)
        
        for _ in range(self.kp_iterations):
            # Generate neighbor by flipping one item
            neighbor = self._flip_item(current)
            neighbor_fitness = self.obj_func(neighbor)
            
            # Accept if improvement
            if neighbor_fitness > current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness
        
        return current
    
    def _flip_item(self, solution: TWDTSPSolution) -> TWDTSPSolution:
        """
        Generate neighbor by flipping one random item
        """
        packing = solution.packing_plan.copy()
        
        # Try to flip a random item
        city_idx = np.random.randint(self.problem.n)
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

    problem = TWDTSPLoader.load_from_file("./cities280/a280_n279_bounded-strongly-corr_01.ttp", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities280/a280_n1395_uncorr-similar-weights_05.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities280/a280_n2790_uncorr_10.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n4460_bounded-strongly-corr_01.ttp.txt", asTTP=True) failed
    #problem = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n22300_uncorr-similar-weights_05.ttp", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities4461/fnl4461_n44600_uncorr_10.ttp.txt", asTTP=True)
    #problem = TWDTSPLoader.load_from_file("./cities33810/pla33810_n33809_bounded-strongly-corr_01.ttp", asTTP=True)


    
    # Run MA2B
    ma2b = MA2B(
        problem=problem,
        pop_size=30,
        num_generations=100,
        tsp_iterations=50,
        kp_iterations=10,
        seed=42
    )
    
    best_solution = ma2b.solve()
    
    print("\n" + "="*50)
    print("MA2B Algorithm - Best Solution Found")
    print("="*50)
    print(best_solution)
    print(f"\nObjective value: {ma2b.best_fitness:.2f}")