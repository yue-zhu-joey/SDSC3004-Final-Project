import numpy as np
from SASS_rand_orth_mat import rand_orth_mat
from SASS_get_r1_r2 import get_r1_r2

def ss(objective_func, dim, pop_size=30, max_iters=1000, bounds=(-100, 100), c=0.8, rp=0.5):
    """
    Simple Spherical Search (SS) Algorithm
    Stripped of adaptive SHADE memory and external archives for pure geometric search.
    
    New Parameters:
    c (float): Fixed Mutation Factor (Step size scaling).
    rp (float): Fixed Crossover Rate (Probability of rotation mask).
    """
    # --- 1. Initialization ---
    x = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    fx = np.array([objective_func.evaluate(ind) for ind in x])
    
    best_idx = np.argmin(fx)
    best_sol = np.copy(x[best_idx])
    best_of = fx[best_idx] 
    
    convergence_curve = np.zeros(max_iters)
    
    # --- 2. Main Optimization Loop ---
    for t in range(max_iters):
        ui = np.zeros((pop_size, dim)) 
        
        # Simple mutually exclusive random indices for r1 and r2
        # Since we removed the archive, popAll size is just pop_size
        r1, r2 = get_r1_r2(pop_size, pop_size)
        
        # Calculate binary diagonal matrix mask using the fixed 'rp' parameter
        mask = np.random.rand(pop_size, dim) < rp 
        # Ensure at least one element comes from the mutant 
        cols = np.random.randint(0, dim, pop_size) 
        mask[np.arange(pop_size), cols] = True 
        
        # Simplified Spherical Search Mutation Base (current-to-best/1)
        zi = best_sol - x + x[r1] - x[r2] 
        
        # Generate one Orthogonal Matrix for this generation
        A = rand_orth_mat(dim, 0.0) 
        
        for i in range(pop_size):
            B = np.diag(mask[i]) 
            # The core Spherical Search geometry: Matrix rotation
            ui[i] = x[i] + c * (zi[i] @ A @ B @ A.T) 
            
        # Boundary handling 
        ui = np.clip(ui, bounds[0], bounds[1])
        
        # Evaluate new population 
        fx_new = np.array([objective_func.evaluate(ind) for ind in ui])
        
        # --- 3. Selection (No memory/archive updates needed) ---
        I = fx_new < fx 
        
        # Replace target vectors with successful trial vectors
        x[I] = ui[I] 
        fx[I] = fx_new[I] 
        
        # Update Global Best Solution
        current_best_idx = np.argmin(fx)
        if fx[current_best_idx] < best_of: 
            best_sol = np.copy(x[current_best_idx]) 
            best_of = fx[current_best_idx] 
            
        convergence_curve[t] = best_of
            
    return best_of, best_sol, convergence_curve