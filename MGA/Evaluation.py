# Import 
import numpy as np
import matplotlib.pyplot as plt
from lambert import LambertIzzo
from TwoBody import two_body_propagator
from scipy.integrate import solve_ivp
from jplephem.calendar import compute_calendar_date

# Integration Tolerance
rtol, atol = 1e-8, 1e-9

def evaluationFunction(x, muS, muV, rV, earth, venus, mars):

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
        plots the trajectory and prints the results
    """

    #***********************************************************************************************************************************************************
    # Unpack all design variables
    #***********************************************************************************************************************************************************
    deptDate    = x[0]     # Departure Date
    leg1TOF     = x[1]     # First leg TOF
    flybyRadius = x[2]*rV  # Flyby Radius
    leg2TOF     = x[3]     # second leg TOF

    flybyDate   = deptDate+leg1TOF     # Epoch of Venus Flyby
    arvDate     = flybyDate+leg2TOF    # Epoch of Mars Arrival 
    
    #***********************************************************************************************************************************************************
    # Leg - 1
    #***********************************************************************************************************************************************************
    earthPosition,earthVelocity = earth.compute_and_differentiate(deptDate)
    venusPosition,venusVelocity = venus.compute_and_differentiate(flybyDate)
    earthVelocity = earthVelocity/86400       # [km/s]
    venusVelocity = venusVelocity/86400       # [km/s]
    vEarthDept, vVenusArv, exitflag = LambertIzzo(muS, earthPosition, venusPosition, leg1TOF, 0)

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

    #***********************************************************************************************************************************************************
    # Flyby Delta-V
    #***********************************************************************************************************************************************************
    
    # Post Flyby Velocity and Venus Departure Velocity should be equal
    J1 = abs(np.linalg.norm(v2_darkside - vVenusDept))
    J2 = abs(np.linalg.norm(v2_sunlitside - vVenusDept))
    Idx = np.argmin([J1, J2])
    
    # Calculate the Delta-v provided by the flyby
    if Idx == 0:
        flybyDelv = abs(np.linalg.norm(v2_darkside - vVenusArv))
        deltaVMisMatch = J1
    else:
        flybyDelv = abs(np.linalg.norm(vVenusArv - v2_sunlitside))
        deltaVMisMatch = J2

    #***********************************************************************************************************************************************************
    # Display Results
    #***********************************************************************************************************************************************************

    launchEpoch     = compute_calendar_date(deptDate)
    flybyEpoch      = compute_calendar_date(flybyDate)
    arrivalEpoch    = compute_calendar_date(arvDate)
    launchDelV      = np.linalg.norm(vEarthDept - earthVelocity)
    arrivalDelV     = np.linalg.norm(vMarsArv - marsVelocity)
    totalDelV       = launchDelV + arrivalDelV

    print('======================== \n')
    print(f'Earth Departure Epoch: {launchEpoch} \n' )
    print(f'Venus Fly-By Epoch: {flybyEpoch} \n')
    print(f'Mars Arrival Epoch: {arrivalEpoch} \n')
    print(f'Launch Delta-V: {launchDelV} km/s \n')
    print(f'Arrival Delta-V: {arrivalDelV} km/s \n')
    print(f'Free Delta-V Achieved through Flyby: {flybyDelv} km/s \n')
    print(f'Mismatch in flyby velocity: {deltaVMisMatch} km/s \n')
    print(f'Total Delta-V: {totalDelV} km/s \n')
    print('------------------------ \n')

    #***********************************************************************************************************************************************************
    # Plot Trajectory
    #***********************************************************************************************************************************************************
    
    # Propagate Celestial Body Orbits
    statesEarth = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, 365*86400), np.hstack((earthPosition, earthVelocity)), rtol=rtol, atol=atol)
    statesVenus = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, 230*86400), np.hstack((venusPosition, venusVelocity)), rtol=rtol, atol=atol)
    statesMars  = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, 687*86400), np.hstack((marsPosition, marsVelocity)), rtol=rtol, atol=atol)

    # Propagate Transfer Trajectory
    statesLeg1  = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, leg1TOF*86400), np.hstack((earthPosition, vEarthDept)), rtol=rtol, atol=atol)
    statesLeg2  = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, leg2TOF*86400), np.hstack((venusPosition, vVenusDept)), rtol=rtol, atol=atol)
    
    # Plot
    fig = plt.figure(figsize=(10, 12)) # width=10 inches, height=6 inches
    ax = fig.add_subplot(111, projection='3d')
    # Planet Orbits
    ax.plot(statesEarth.y[0], statesEarth.y[1], statesEarth.y[2], 'b-', label='Earth Orbit')
    ax.plot(statesVenus.y[0], statesVenus.y[1], statesVenus.y[2], 'y-', label='Venus Orbit')
    ax.plot(statesMars.y[0], statesMars.y[1], statesMars.y[2], 'r-', label='Mars Orbit')
    # Points 
    ax.scatter(earthPosition[0], earthPosition[1], earthPosition[2], color='b', marker='o', label='Departure')
    ax.scatter(venusPosition[0], venusPosition[1], venusPosition[2], color='k', marker='o', label='Fly-By')
    ax.scatter(marsPosition[0], marsPosition[1], marsPosition[2], color='r', marker='o', label='Fly-By')
    # Transfer Trajectory
    ax.plot(statesLeg1.y[0], statesLeg1.y[1], statesLeg1.y[2], 'b-', label='Pre-FlyBy Trajectory')
    ax.plot(statesLeg2.y[0], statesLeg2.y[1], statesLeg2.y[2], 'y-', label='Post-FlyBy Trajectory')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Earth-Venus Trajectory')
    ax.legend(loc='upper center', mode='expand', numpoints=1, ncol=4, fancybox = True, fontsize='small')
    ax.grid(True)
    plt.show()
  