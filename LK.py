import numpy as np
from typing import List

class LKSolver:
    def __init__(self, coords: np.ndarray):
        self.instance = coords
        self.candidates = self._precompute_candidates() 

    def _precompute_candidates(self, max_neighbors=20):
        """
        Precompute k-nearest for all cities
        """
        n = len(self.instance)
        candidates = [[] for _ in range(n)]
    
        for i in range(n):
            distances = np.linalg.norm(self.instance - self.instance[i], axis=1)
            idx = np.argsort(distances)[1:max_neighbors+1]
            candidates[i] = idx.tolist()
    
        return candidates

    def _cost(self, i, j):
        """
        Euclidian distance between cities
        """
        return np.linalg.norm(self.instance[i] - self.instance[j])

    def _tour_cost(self, tour):
        """
        Calculate total tour cost
        """
        n = len(tour)
        return sum(self._cost(tour[i], tour[(i+1) % n]) for i in range(n))

    def _neighbor(self, city):
        """
        20 nearest neighbors to city (LK efficiency trick)
        """
        return self.candidates[city]
    
    def _initialTour(self):
        n = len(self.instance)
        tour = [0]
        unvisited = set(range(1, n))
    
        while unvisited:
            last = tour[-1]
            # Only compute distances from LAST city to unvisited
            dist_to_unvisited = {city: np.linalg.norm(self.instance[last] - self.instance[city]) 
                                 for city in unvisited}
            nearest = min(dist_to_unvisited, key=dist_to_unvisited.get)
            tour.append(nearest)
            unvisited.remove(nearest)
    
        return tour
    
    def _perturbation_LK(self, tour):
        """
        Double-bridge perturbation (LKH-style diversification)
        """
        n = len(tour)
        if n < 8:
            return tour[:]
            
        idx = sorted(np.random.choice(n, 4, replace=False))
        
        # Double bridge: break at 4 points and reconnect differently
        a, b, c, d = idx
        new_tour = tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]
        
        return new_tour

    def _two_opt(self, tour):
        """
        Simple 2-opt local search for improvement
        """
        n = len(tour)
        improved = True
        
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # Calculate change in tour length
                    # Current: (i, i+1) and (j, j+1)
                    # New: (i, j) and (i+1, j+1)
                    curr_dist = self._cost(tour[i], tour[i+1]) + self._cost(tour[j], tour[(j+1) % n])
                    new_dist = self._cost(tour[i], tour[j]) + self._cost(tour[i+1], tour[(j+1) % n])
                    
                    if new_dist < curr_dist:
                        # Reverse segment between i+1 and j
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        break
                if improved:
                    break
                    
        return tour

    def _try_lk_move(self, tour, t1):
        """
        Try to find an improving LK move starting from city t1
        Returns (improved_tour, gain) or (None, 0) if no improvement
        """
        n = len(tour)
        t1_idx = tour.index(t1)
        
        # Try breaking edge (t1, t2) where t2 = successor of t1
        t2_idx = (t1_idx + 1) % n
        t2 = tour[t2_idx]
        
        # Look for a better connection from t1
        for t3 in self._neighbor(t1):
            if t3 == t2:  # Skip current successor
                continue
                
            # Calculate gain from swapping (t1,t2) with (t1,t3)
            old_dist = self._cost(t1, t2)
            new_dist = self._cost(t1, t3)
            
            if new_dist >= old_dist:  # No improvement
                continue
            
            # Find t3 in tour and try to close the tour
            t3_idx = tour.index(t3)
            
            # Try 2-opt style reconnection
            # Reverse the segment between t2 and t3
            if t2_idx < t3_idx:
                new_tour = tour[:t2_idx] + tour[t2_idx:t3_idx+1][::-1] + tour[t3_idx+1:]
            else:
                new_tour = tour[:t3_idx+1] + tour[t3_idx+1:t2_idx][::-1] + tour[t2_idx:]
            
            # Check if this actually improves the tour
            new_cost = self._tour_cost(new_tour)
            old_cost = self._tour_cost(tour)
            
            if new_cost < old_cost:
                return new_tour, old_cost - new_cost
                
        return None, 0

    def _localSearch_LK(self, tour: List[int]):
        """
        Perform local search using simplified LK-style moves
        """
        improvement = True
        iterations = 0
        max_iterations = 100
        
        while improvement and iterations < max_iterations:
            improvement = False
            iterations += 1
            
            for t1 in tour:
                new_tour, gain = self._try_lk_move(tour, t1)
                
                if gain > 0:
                    tour = new_tour
                    improvement = True
                    break
                    
            # Also try 2-opt for thoroughness
            if not improvement:
                old_cost = self._tour_cost(tour)
                tour = self._two_opt(tour)
                new_cost = self._tour_cost(tour)
                if new_cost < old_cost:
                    improvement = True
                    
        return tour

    def lk(self, max_iter):
        """
        Lin-Kernighan heuristic for TSP with iterated local search
        """
        # Get initial tour
        T = self._initialTour()
        T = self._localSearch_LK(T)
        
        best_tour = T[:]
        best_cost = self._tour_cost(T)
        
        print(f"Initial tour cost: {best_cost:.2f}")
        
        # Iterated local search
        no_improvement = 0
        for iter_count in range(max_iter):
            # Perturb current best
            T_prime = self._perturbation_LK(best_tour)
            T_prime = self._localSearch_LK(T_prime)
            
            cost_prime = self._tour_cost(T_prime)
            
            if cost_prime < best_cost:
                best_tour = T_prime
                best_cost = cost_prime
                no_improvement = 0
                print(f"Iter {iter_count}: New best cost = {best_cost:.2f}")
            else:
                no_improvement += 1
                
            # Early stopping if no improvement for a quarter of a while
            if no_improvement > max_iter // 4:
                break
        
        best_tour = [city + 1 for city in best_tour] # returning to 1 indexed cities
        return best_tour


def main():
    """Test the LK TSP solver"""
    
    # Generate test instance (50 cities in unit square)
    np.random.seed(42)
    n_cities = 50
    coords = np.random.rand(n_cities, 2)
    
    # Create solver
    solver = LKSolver(coords)
    
    print(f"Solving {n_cities}-city TSP instance...")
    print("-" * 50)
    
    # Solve with LK
    tour = solver.lk(max_iter=50)
    
    # Compute tour cost
    total_cost = solver._tour_cost(tour)
    
    print("-" * 50)
    print(f"Final tour length: {total_cost:.2f}")
    print(f"First 10 cities: {tour}")
    
    # Compare with naive nearest neighbor
    nn_tour = solver._initialTour()
    nn_cost = solver._tour_cost(nn_tour)
    improvement = ((nn_cost - total_cost) / nn_cost) * 100
    
    print(f"NN baseline:     {nn_cost:.2f}")
    print(f"LK improvement:  {improvement:.1f}%")
    
    # Verify it's a valid tour
    assert len(set(tour)) == n_cities, "Tour doesn't visit all cities!"
    assert len(tour) == n_cities, "Wrong tour length!"
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    main()