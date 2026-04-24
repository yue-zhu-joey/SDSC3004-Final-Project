import numpy as np

def run_differential_evolution(objective_func, dim, bounds, pop_size=30, max_fes=300000, F=0.8, CR=0.9):
    """
    Standard Differential Evolution (DE/rand/1/bin).

    Returns:
        best_solution: best vector found
        best_fitness: best objective value
        fitness_history: list of best fitness values by generation
    """
    lower_bound, upper_bound = bounds
    
    # 1. INITIALIZATION
    # Generate initial population randomly within the [-100, 100] bounds
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dim))
    
    # Evaluate initial population
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = objective_func.evaluate(population[i])
    
    fes = pop_size # Track function evaluations
    
    # Track the global best
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Optional: Keep track of the best score over time for your final report graphs
    fitness_history = [best_fitness]

    # MAIN OPTIMIZATION LOOP
    while fes < max_fes:
        for i in range(pop_size):
            # The current individual is the "Target Vector"
            target_vector = population[i]
            
            # 2. MUTATION
            # Select 3 distinct random individuals that are NOT the current index 'i'
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            
            # Create the "Donor Vector"
            # F_dynamic = np.random.uniform(0.5, 1.0)
            
            donor_vector = target_vector + F * (best_solution - target_vector) + F * (population[r1] - population[r2])
            
            # Boundary Control: If mutation pushes the vector outside [-100, 100], clip it back
            donor_vector = np.clip(donor_vector, lower_bound, upper_bound)
            
            # 3. CROSSOVER (Binomial)
            # Create the "Trial Vector" by mixing Target and Donor
            cross_points = np.random.rand(dim) <= CR
            
            # Ensure at least one dimension is inherited from the donor vector
            j_rand = np.random.randint(0, dim)
            cross_points[j_rand] = True 
            
            # np.where works like: if cross_point is True, take donor, else take target
            trial_vector = np.where(cross_points, donor_vector, target_vector)
            
            # 4. SELECTION
            # Evaluate the new trial vector
            trial_fitness = objective_func.evaluate(trial_vector)
            fes += 1
            
            # Greedy selection: If the new vector is better or equal, it survives
            if trial_fitness <= fitness[i]:
                population[i] = trial_vector
                fitness[i] = trial_fitness
                
                # Update global best if a new absolute minimum is found
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_vector.copy()
            
            # Stop exactly when we hit the MaxFEs limit
            if fes >= max_fes:
                break
                
        # Record the best fitness at the end of this generation
        fitness_history.append(best_fitness)
        
    return best_solution, best_fitness, fitness_history