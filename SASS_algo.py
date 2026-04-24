import numpy as np
from SASS_rand_orth_mat import rand_orth_mat
from SASS_get_r1_r2 import get_r1_r2

def sass(objective_func, dim, pop_size=30, max_iters=1000, bounds=(-100, 100)):
    """
    Unconstrained Spherical Search (SASS) Algorithm
    Translated from the provided MATLAB scripts.
    """
    # --- 1. Initialization ---
    x = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    fx = np.array([objective_func.evaluate(ind) for ind in x])
    
    # SHADE Historical Memory Setup
    memory_size = 5 
    archive_c = np.ones(memory_size) * 0.5 # mu_c 
    archive_rp = np.ones(memory_size) * 0.5 # mu_rp 
    hist_pos = 0 
    
    # External Archive for failed parents
    archive = [] 
    
    best_idx = np.argmin(fx)
    best_sol = np.copy(x[best_idx])
    best_of = fx[best_idx] 
    
    convergence_curve = np.zeros(max_iters)
    
    # --- 2. Main Optimization Loop ---
    for t in range(max_iters):
        ui = np.zeros((pop_size, dim)) 
        
        # Determine memory index for each individual
        mem_rand_index = np.random.randint(0, memory_size, pop_size) 
        mu_c = archive_c[mem_rand_index] 
        mu_rp = archive_rp[mem_rand_index] 
        
        # Generate rp (Crossover Rate) using Normal Distribution approximation 
        rp = mu_rp + 0.1 * np.random.normal(size=pop_size)
        rp[mu_rp == -1] = 0 
        rp = np.clip(rp, 0, 1)
        
        # Generate c (Mutation Factor) using Cauchy Distribution
        c = mu_c + 0.1 * np.random.standard_cauchy(size=pop_size)
        # Resample values <= 0 
        while np.any(c <= 0):
            pos = c <= 0 
            c[pos] = mu_c[pos] + 0.1 * np.random.standard_cauchy(size=np.sum(pos)) 
        c = np.minimum(c, 1)
        
        # Setup population + archive for r2 selection
        popAll = np.vstack((x, archive)) if len(archive) > 0 else x 
        r1, r2 = get_r1_r2(pop_size, len(popAll))
        
        # Calculation of pbest solution
        pNP = max(int(round(0.1 * pop_size)), 2) 
        sorted_idx = np.argsort(fx)
        randindex = np.random.randint(0, pNP, pop_size) 
        phix = x[sorted_idx[randindex]] 
        
        # Calculate binary diagonal matrix mask
        mask = np.random.rand(pop_size, dim) < rp[:, None] 
        # Ensure at least one element comes from the mutant 
        cols = np.random.randint(0, dim, pop_size) 
        mask[np.arange(pop_size), cols] = True 
        
        # Spherical Search Mutation Base
        zi = phix - x + x[r1] - popAll[r2] 
        
        # Binary Orthogonal Matrix application
        A = rand_orth_mat(dim, 0.0) 
        for i in range(pop_size):
            B = np.diag(mask[i]) 
            # Matrix multiplication equivalent to MATLAB's A*B*A'
            ui[i] = x[i] + c[i] * (zi[i] @ A @ B @ A.T) 
            
        # Boundary handling 
        ui = np.clip(ui, bounds[0], bounds[1])
        
        # Evaluate new population 
        fx_new = np.array([objective_func.evaluate(ind) for ind in ui])
        
        # --- 3. Memory & Archive Updates ---
        diff = np.abs(fx - fx_new) 
        I = fx_new < fx 
        
        goodRP = rp[I] 
        goodC = c[I] 
        
        if np.sum(I) > 0:
            # Update Archive 
            for ind in x[I]:
                archive.append(ind.copy())
            # Maintain archive size limit (typically pop_size)
            if len(archive) > pop_size:
                indices_to_keep = np.random.choice(len(archive), pop_size, replace=False)
                archive = [archive[idx] for idx in indices_to_keep]
                
            # Update Memory C and RP using Weighted Lehmer Mean [cite: 111, 113]
            num_success_params = len(goodRP) 
            if num_success_params > 0: 
                weightsSS = diff[I] / np.sum(diff[I]) 
                
                archive_c[hist_pos] = np.sum(weightsSS * (goodC ** 2)) / np.sum(weightsSS * goodC) 
                
                if np.max(goodRP) == 0 or archive_rp[hist_pos] == -1: 
                    archive_rp[hist_pos] = -1 
                else:
                    archive_rp[hist_pos] = np.sum(weightsSS * (goodRP ** 2)) / np.sum(weightsSS * goodRP) 
                    
                hist_pos = hist_pos + 1 
                if hist_pos >= memory_size: 
                    hist_pos = 0 
                    
        # Replace target vectors with successful trial vectors [cite: 118, 119]
        x[I] = ui[I] 
        fx[I] = fx_new[I] 
        
        # Update Global Best Solution
        current_best_idx = np.argmin(fx)
        if fx[current_best_idx] < best_of: 
            best_sol = np.copy(x[current_best_idx]) 
            best_of = fx[current_best_idx] 
            
        convergence_curve[t] = best_of
        
        # if (t + 1) % 100 == 0:
        #     print(f"Iteration {t+1}/{max_iters} | Best Score: {best_of:.4f}")
            
    return best_of, best_sol, convergence_curve