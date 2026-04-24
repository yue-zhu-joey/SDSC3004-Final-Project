import numpy as np

def run_jde(objective_func, dim, bounds, pop_size=30, max_fes=300000, trial_prob=0.1):
    """
    Self-Adaptive Differential Evolution (jDE) with Training Log
    """
    lower_bound, upper_bound = bounds
    
    # 1. INITIALIZATION
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = objective_func.evaluate(population[i])
        
    fes = pop_size 
    
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    convergence_curve = [best_fitness]
    
    # --- LOGGING: Print Initial State ---
    # print(f"Starting optimization... Initial Best Fitness: {best_fitness:.4f}")
    
    # --- jDE UPGRADE: Initialize F and CR arrays for every individual ---
    F_array = np.ones(pop_size) * 0.5
    CR_array = np.ones(pop_size) * 0.9
    
    generation = 0
    
    # MAIN OPTIMIZATION LOOP
    while fes < max_fes:
        generation += 1
        
        for i in range(pop_size):
            target_vector = population[i]
            
            # --- jDE UPGRADE: 10% chance to evolve parameters before mutating ---
            F_trial = F_array[i]
            CR_trial = CR_array[i]
            
            if np.random.rand() < trial_prob:
                F_trial = 0.1 + np.random.rand() * 0.9  # New F between 0.1 and 1.0
            if np.random.rand() < trial_prob:
                CR_trial = np.random.rand()             # New CR between 0.0 and 1.0
            
            # 2. MUTATION (current-to-best/1)
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2 = np.random.choice(indices, 2, replace=False)
            
            # Use the newly evolved F_trial
            donor_vector = target_vector + F_trial * (best_solution - target_vector) + F_trial * (population[r1] - population[r2])
            donor_vector = np.clip(donor_vector, lower_bound, upper_bound)
            
            # 3. CROSSOVER
            # Use the newly evolved CR_trial
            cross_points = np.random.rand(dim) <= CR_trial
            j_rand = np.random.randint(0, dim)
            cross_points[j_rand] = True 
            
            trial_vector = np.where(cross_points, donor_vector, target_vector)
            
            # 4. SELECTION
            trial_fitness = objective_func.evaluate(trial_vector)
            fes += 1
            
            # If the trial is better, it survives AND it keeps its new parameters
            if trial_fitness <= fitness[i]:
                population[i] = trial_vector
                fitness[i] = trial_fitness
                F_array[i] = F_trial   # Save successful F
                CR_array[i] = CR_trial # Save successful CR
                
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_vector.copy()
            
            if fes >= max_fes:
                break
                
        convergence_curve.append(best_fitness)
        
        # --- LOGGING: Print progress every 100 generations ---
        # if generation % 100 == 0:
        #     print(f"Gen: {generation:4d} | FEs: {fes:5d}/{max_fes} | Best Fitness: {best_fitness:.6f}")
            
    # --- LOGGING: Print Final State ---
    # print(f"Optimization Finished! Final Best Fitness: {best_fitness:.6f}\n")
        
    return best_solution, best_fitness, convergence_curve