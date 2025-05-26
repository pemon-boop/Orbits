# Import 
import numpy as np
from lambert import LambertIzzo


def costFunction(gaInstance, x, x_idx, muS, muM, rM, earth, mars):
    """
    This function creates a Heliocentric trajectory connecting Earth and Mars

    Inputs:
        x             - GA design variables for mga-0 DSM model
        muS           - Gravitational Parameter of Sun
        muM           - Gravitational Parameter of Mars
        rV            - Radius of Mars
        earth         - Ephemeris of Earth
        mars          - Ephemeris of Mars

    Outputs:
        J             - cost function for a rendezvous at the target planet
    """

    # ***********************************************************************************************************************************************************
    # Unpack all design variables
    # ***********************************************************************************************************************************************************
    deptDate = abs(x[0])  # Departure Date [JD]
    leg1TOF = abs(x[1])  # First leg TOF  [days]

    arvDate = deptDate + leg1TOF  # Epoch of Mars Arrival [JD]

    # ***********************************************************************************************************************************************************
    # Leg - 1
    # ***********************************************************************************************************************************************************
    try:
        earthPosition, earthVelocity = earth.compute_and_differentiate(deptDate)
        earthVelocity = earthVelocity/86400
        marsPosition, marsVelocity = mars.compute_and_differentiate(arvDate)
        marsVelocity = marsVelocity/86400

        vEarthDept, vMarsArv, exitflag = LambertIzzo(muS, earthPosition, marsPosition, leg1TOF, 0)  # Use 0 revs first
        
        # If Lambert didn't converge, try with 1 revolution
        if exitflag == -1:
            vEarthDept, vMarsArv, exitflag = LambertIzzo(muS, earthPosition, marsPosition, leg1TOF, 1)
            
        # If still didn't work, penalize this solution
        if (exitflag == -1) or (exitflag == -2):
            return 1e-5  # Very small fitness (remember we're maximizing 1/cost)
    
    except Exception as e:
        # Handle any other exceptions (e.g., out of bounds for ephemeris)
        return 1e-5  # Very small fitness for invalid solutions

    # ***********************************************************************************************************************************************************
    # Cost Function - Multi-component objective
    # ***********************************************************************************************************************************************************

    # 1. Launch delta-V (Earth departure)
    launch_dv = np.linalg.norm(vEarthDept - earthVelocity)
    
    # 2. Arrival delta-V (Mars arrival)
    arrival_dv = np.linalg.norm(vMarsArv - marsVelocity)
    
    # 3. Total delta-V
    total_dv = launch_dv + arrival_dv
    
    # Optional: Add penalty for extremely short/long transfers
    tof_penalty = 0
    # if leg1TOF < 180:  # Too short transfers might be unrealistic
    #     tof_penalty = 10 * (180 - leg1TOF)
    # elif leg1TOF > 900:  # Longer transfers increase mission risks
    #     tof_penalty = 0.1 * (leg1TOF - 900)
    
    # # Combined cost function with weights
    J = total_dv + tof_penalty
    
    # Compute generation statistics for logging (if this is a new generation)
    current_gen = gaInstance.generations_completed
    if hasattr(gaInstance, 'last_recorded_gen') and gaInstance.last_recorded_gen != current_gen:
        # Get all fitness values of current population
        population_fitness = gaInstance.last_generation_fitness
        if population_fitness is not None and len(population_fitness) > 0:
            valid_fitness = [f for f in population_fitness if f > 0]
            if valid_fitness:
                best_dv = 1.0/max(valid_fitness) - 1e-8
                print(f"Generation {current_gen}: Best ΔV = {best_dv:.2f} km/s")
    
    # Store current generation
    if not hasattr(gaInstance, 'last_recorded_gen'):
        gaInstance.last_recorded_gen = 0
    gaInstance.last_recorded_gen = current_gen
    
    # In PyGAD, we're maximizing, so return 1/J (add small constant to avoid division by zero)
    return 1.0 / (J + 1e-8)