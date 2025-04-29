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
    earthPosition, earthVelocity = earth.compute_and_differentiate(deptDate)
    earthVelocity = earthVelocity/86400
    marsPosition, marsVelocity = mars.compute_and_differentiate(arvDate)
    marsVelocity = marsVelocity/86400

    vEarthDept, vMarsArv, exitflag = LambertIzzo(muS, earthPosition, marsPosition, leg1TOF, 0)

    # If Lambert didnt converge return a high np.cost
    if (exitflag == -1) or (exitflag == -2):
        J = 200000
        return J

    # ***********************************************************************************************************************************************************
    # Cost Function
    # ***********************************************************************************************************************************************************

    J = abs(np.linalg.norm(vEarthDept - vMarsArv))

    # In Python GA tries to maximize, thus we take the inverse to minimize
    J = 1.0 / (J + 1e-8)

    return J
