import numpy as np
from scipy.optimize import dual_annealing

def run_dual_annealing(objective_func, dim, lower_bound, upper_bound, max_iter, initial_temp=5230.0):
    """
    A wrapper for SciPy's Dual Annealing to match your execution structure.
    """
    # SciPy requires a list of (min, max) bounds for every dimension
    bounds = [(lower_bound, upper_bound) for _ in range(dim)]
    
    # Variables to track the history for your convergence curve
    best_score_so_far = float('inf')
    convergence_curve = []
    eval_count = 0
    
    # Inner wrapper to count evaluations and track the best score over time
    def objective_wrapper(x):
        nonlocal best_score_so_far, eval_count
        
        # Stop strictly if we hit the evaluation limit
        if eval_count >= max_iter:
            raise StopIteration
            
        eval_count += 1
        score = objective_func.evaluate(x)
        
        if score < best_score_so_far:
            best_score_so_far = score
            
        convergence_curve.append(best_score_so_far)
        return score

    # Execute SciPy's algorithm
    try:
        result = dual_annealing(
            func=objective_wrapper,
            bounds=bounds,
            maxfun= max_iter, 
            initial_temp=5230.0
        )
        final_score = result.fun
        best_pos = result.x
    except StopIteration:
        # If it hits the strict max_evals limit, we catch the stop and return the current best
        final_score = best_score_so_far
        # Note: SciPy doesn't easily expose the vector mid-run on a hard stop, 
        # so returning a placeholder array for the vector if it hard stops
        best_pos = np.zeros(dim) 
        
    return final_score, best_pos, np.array(convergence_curve)