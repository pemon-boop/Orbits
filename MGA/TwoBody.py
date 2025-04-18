# Import 
import numpy as np 

def two_body_propagator(t, y, mu):
    """
    Computes the time derivative of the state vector y for the two-body problem.
    
    Parameters:
      t  (float)    : Time 
      y  (ndarray)  : state vector [r_x, r_y, r_z, v_x, v_y, v_z].
      mu (float)    : gravitational parameter
    
    Returns:
      dydt (ndarray) : The time derivative [v, a].
    """
    r = y[:3]
    v = y[3:]
    a = -mu * r / np.linalg.norm(r)**3
    return np.concatenate((v, a))