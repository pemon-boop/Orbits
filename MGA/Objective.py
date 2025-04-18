# Import 
import numpy as np
from lambert import LambertIzzo

def costFunction(gaInstance, x, x_idx, muS, muV, rV, earth, venus, mars):

    """
    This function creates a Heliocentric trajectory connecting Earth and Mars using a Venus flyby 
    
    Inputs:
        x             - GA design variables for mga-0 DSM model
        muS           - Gravitational Parameter of Sun 
        muV           - Gravitational Parameter of Venus
        rV            - Radius of Venus
        earth         - Ephemeris of Earth
        venus         - Ephemeris of Venus
        mars          - Ephemeris of Mars
    
    Outputs:
        J             - cost function for a rendezvous at the target planet
    """

    #***********************************************************************************************************************************************************
    # Unpack all design variables
    #***********************************************************************************************************************************************************
    deptDate    = abs(x[0])     # Departure Date [JD]
    leg1TOF     = abs(x[1])     # First leg TOF  [days]
    flybyRadius = abs(x[2])*rV  # Flyby Radius   [fraction] 
    leg2TOF     = abs(x[3])     # second leg TOF [days]

    flybyDate   = deptDate+leg1TOF     # Epoch of Venus Flyby  [JD]
    arvDate     = flybyDate+leg2TOF    # Epoch of Mars Arrival [JD]
    
    #***********************************************************************************************************************************************************
    # Leg - 1
    #***********************************************************************************************************************************************************
    earthPosition,earthVelocity = earth.compute_and_differentiate(deptDate)
    venusPosition,venusVelocity = venus.compute_and_differentiate(flybyDate)
    earthVelocity = earthVelocity/86400       # [km/s]
    venusVelocity = venusVelocity/86400       # [km/s]
    vEarthDept, vVenusArv, exitflag = LambertIzzo(muS, earthPosition, venusPosition, leg1TOF, 0)
    
    # If Lambert didnt converge return a high np.cost 
    if (exitflag == -1) or (exitflag == -2):
        J = 1.0 / (2000 + 1e-8)
        return J
    
    #***********************************************************************************************************************************************************
    # Gravity Assist
    #***********************************************************************************************************************************************************
    
    # unit vectors in the Sun direction and Venus' velocity direction
    uSun = -venusPosition/np.linalg.norm(venusPosition)   # opp to the direction of the planet from sun
    uVenus = venusVelocity/np.linalg.norm(venusVelocity)

    # V-Inf in bound
    vInf1 = vVenusArv - venusVelocity
    
    # Calculate the deflection angle
    vInf1_Sun   = np.dot(vInf1,uSun)                 # V-infinity projected on the sun direction
    vInf1_V     = np.dot(vInf1,uVenus)               # V-infinity projected on the planet's velocity direction
    vInf1       = vInf1_V*uVenus + vInf1_Sun*uSun    # Calculate inbound V-infinity 
    VInfMag     = np.linalg.norm(vInf1)              #  Magnitude of V-infinity 
    
    # Calculate angle Phi between V_planet and in bound V_inf_1
    phi1 = np.arctan2(vInf1_Sun, vInf1_V)
    
    # Gravity Assist Deflection Angle
    e = 1 + flybyRadius*VInfMag**2/muV
    d = 2*np.arcsin(1/e)
    
    # Dark Side Approach 
    phi2 = phi1+d
    vInf2_darkside = VInfMag*np.cos(phi2)*uVenus + VInfMag*np.sin(phi2)*uSun     # outbound V_infinity
    v2_darkside = venusVelocity + vInf2_darkside                           # outbound heliocentric velocity

    # Sun-Lit Side 
    phi2 = phi1-d 
    vInf2_sunlitside = VInfMag*np.cos(phi2)*uVenus + VInfMag*np.sin(phi2)*uSun   # Calculate outbound V_infinity_2
    v2_sunlitside = venusVelocity + vInf2_sunlitside                       # Outbound velocity vector (lets call it v2 convenience)

    #***********************************************************************************************************************************************************
    # Leg - 2
    #***********************************************************************************************************************************************************

    marsPosition,marsVelocity = mars.compute_and_differentiate(arvDate)
    marsVelocity = marsVelocity/86400  # [km/s]
    vVenusDept, vMarsArv, exitflag = LambertIzzo(muS, venusPosition, marsPosition, leg2TOF, 0)
    
    # If Lambert didnt converge return a high np.cost 
    if (exitflag == -1) or (exitflag == -2):
        J = 1.0 / (1000 + 1e-8)
        return J
    
    #***********************************************************************************************************************************************************
    # Cost Function
    #***********************************************************************************************************************************************************
    
    # Post Flyby Velocity and Venus Departure Velocity should be equal
    J1 = abs(np.linalg.norm(v2_darkside - vVenusDept))
    J2 = abs(np.linalg.norm(v2_sunlitside - vVenusDept))
    J = min(J1,J2)

    # In Python GA tries to maximize, thus we take the inverse to minimize
    J = 1.0 / (J + 1e-8)

    return J
