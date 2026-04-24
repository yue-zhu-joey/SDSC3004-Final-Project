import numpy as np
from SASS_rand_orth_mat import rand_orth_mat
from SASS_get_r1_r2 import get_r1_r2

def l_sass(objective_func, dim, pop_size=100, bounds=(-100, 100), max_fes=30000):
    """
    L-SASS: Spherical Search Algorithm with Linear Population Size Reduction
    Fixed to correctly synchronize dynamic population shrinking.
    """
    # --- CRITICAL FIX: Freeze initial population size for the math formula ---
    pop_size = pop_size  
    
    lower_bound, upper_bound = bounds
    pop_min = 4  # The absolute minimum population size required for DE math
    
    # --- 1. INITIALIZATION ---
    x = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    fx = np.array([objective_func.evaluate(ind) for ind in x])
    
    fes = pop_size
    
    # SHADE Historical Memory Setup
    memory_size = 5 
    archive_c = np.ones(memory_size) * 0.5 
    archive_rp = np.ones(memory_size) * 0.5 
    hist_pos = 0 
    
    # External Archive for failed parents
    archive = [] 
    
    best_idx = np.argmin(fx)
    best_sol = np.copy(x[best_idx])
    best_of = fx[best_idx] 
    
    convergence_curve = [best_of]
    generation = 0
    
    print(f"Starting L-SASS optimization... Initial Best Fitness: {best_of:.4f}")
    
    # --- 2. MAIN OPTIMIZATION LOOP ---
    while fes < max_fes:
        generation += 1
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
        cols = np.random.randint(0, dim, pop_size) 
        mask[np.arange(pop_size), cols] = True 
        
        # Spherical Search Mutation Base
        zi = phix - x + x[r1] - popAll[r2] 
        
        # Binary Orthogonal Matrix application
        A = rand_orth_mat(dim, 0.0) 
        for i in range(pop_size):
            B = np.diag(mask[i]) 
            ui[i] = x[i] + c[i] * (zi[i] @ A @ B @ A.T) 
            
        # Boundary handling 
        ui = np.clip(ui, bounds[0], bounds[1])
        
        # Evaluate new population 
        fx_new = np.zeros(pop_size)
        for i in range(pop_size):
            if fes >= max_fes:
                fx_new[i] = float('inf') # Skip if budget is blown
            else:
                fx_new[i] = objective_func.evaluate(ui[i])
                fes += 1
        
        # --- 3. Memory & Archive Updates ---
        diff = np.abs(fx - fx_new) 
        I = fx_new < fx 
        
        goodRP = rp[I] 
        goodC = c[I] 
        
        if np.sum(I) > 0:
            # Update Archive 
            for ind in x[I]:
                archive.append(ind.copy())
            # Maintain archive size limit (scales with current pop_size)
            if len(archive) > pop_size:
                indices_to_keep = np.random.choice(len(archive), pop_size, replace=False)
                archive = [archive[idx] for idx in indices_to_keep]
                
            # Update Memory C and RP using Weighted Lehmer Mean 
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
                    
        # Replace target vectors with successful trial vectors 
        x[I] = ui[I] 
        fx[I] = fx_new[I] 
        
        # Update Global Best Solution
        current_best_idx = np.argmin(fx)
        if fx[current_best_idx] < best_of: 
            best_sol = np.copy(x[current_best_idx]) 
            best_of = fx[current_best_idx] 
            
        convergence_curve.append(best_of)
        
        # --- 4. LINEAR POPULATION SIZE REDUCTION (LPSR) ---
        # Calculate what the population size should be right now
        new_pop_size = int(round(((pop_min - pop_size) / max_fes) * fes) + pop_size)
        new_pop_size = max(new_pop_size, pop_min) # Never drop below 4
        
        if new_pop_size < pop_size:
            # Sort the current population to find the worst vectors
            sort_idx = np.argsort(fx)
            
            # Slice the arrays, keeping only the best 'new_pop_size' individuals
            x = x[sort_idx[:new_pop_size]]
            fx = fx[sort_idx[:new_pop_size]]
            
            # Synchronize the active pop_size variable with the newly sliced matrices
            pop_size = new_pop_size
            
            # Trim the external archive if it is now larger than the new population size
            if len(archive) > pop_size:
                indices_to_keep = np.random.choice(len(archive), pop_size, replace=False)
                archive = [archive[idx] for idx in indices_to_keep]
        
        # Logging progress
        if generation % 100 == 0:
            print(f"Gen: {generation:4d} | FEs: {fes:5d}/{max_fes} | Pop: {pop_size:3d} | Best Score: {best_of:.4f}")
            
    print(f"Optimization Finished! Final Best Fitness: {best_of:.6f}\n")
    return best_of, best_sol, convergence_curve