import numpy as np

def get_r1_r2(pop_size, pop_all_size):
    """
    Python equivalent of gnR1R2. 
    Generates mutually exclusive random indices efficiently.
    """
    r1 = np.zeros(pop_size, dtype=int)
    r2 = np.zeros(pop_size, dtype=int)
    for i in range(pop_size):
        # r1 must not equal i
        r1_choices = [x for x in range(pop_size) if x != i]
        r1[i] = np.random.choice(r1_choices)
        
        # r2 must not equal i or r1[i], drawn from the combined population+archive
        r2_choices = [x for x in range(pop_all_size) if x != i and x != r1[i]]
        r2[i] = np.random.choice(r2_choices)
    return r1, r2