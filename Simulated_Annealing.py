# Simulated Anealing
import numpy as np

def simulated_annealing(objective_func, dim, lower_bound, upper_bound, initial_temp, cooling_rate, max_iter):
    """Run simulated annealing and return final score, best vector, and history.

    The function treats max_iter as the maximum number of objective evaluations.
    """

    # initialization
    current_sol = np.random.uniform(lower_bound, upper_bound, dim)
    current_score = objective_func.evaluate(current_sol)
    T = initial_temp
    history = [current_score]
    eval_count = 1

    min_temp = 1e-12
    while eval_count < max_iter:
        # neighborhood search
        noise = np.random.normal(loc=0, scale=5.0, size=dim)
        neighbor_sol = current_sol + noise

        # Force the neighbor solution to be within bounds
        neighbor_sol = np.clip(neighbor_sol, lower_bound, upper_bound)
        neighbor_score = objective_func.evaluate(neighbor_sol)
        eval_count += 1

        # Calculate the score difference
        delta_score = neighbor_score - current_score

        if delta_score < 0:
            # It's a better solution, accept it
            current_sol = neighbor_sol
            current_score = neighbor_score
        else:
            # It's a worse solution, accept it with a probability.
            # Use a minimum temperature to prevent overflow in division.
            safe_T = max(T, min_temp)
            exponent = -delta_score / safe_T
            if exponent <= -745:
                probability = 0.0
            elif exponent >= 709:
                probability = 1.0
            else:
                probability = np.exp(exponent)

            # Flip a weighted coin to decide whether to accept the worse solution
            if np.random.rand() < probability:
                current_sol = np.copy(neighbor_sol)
                current_score = neighbor_score

        history.append(current_score)
        # Cool down the temperature
        T *= cooling_rate
        T = max(T, min_temp)

    return current_score, current_sol, history



