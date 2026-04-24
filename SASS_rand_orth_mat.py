import numpy as np

def rand_orth_mat(n, t=0.0):
    R = np.eye(n)
    # l = randperm(n) equivalent
    l = np.random.permutation(n)  # Generate a random permutation of indices from 0 to n-1
    
    for ii in range(n // 2):
        i = 2 * ii # 0-indexed adjustment
        R[l[i], l[i]] = np.sin(t)  
        R[l[i+1], l[i+1]] = np.sin(t)  
        R[l[i], l[i+1]] = np.cos(t)  
        R[l[i+1], l[i]] = -np.cos(t) 
    return R