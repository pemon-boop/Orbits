# Import 
import numpy as np
import matplotlib.pyplot as plt
from lambert import LambertIzzo
from TwoBody import two_body_propagator
from scipy.integrate import solve_ivp
from jplephem.calendar import compute_calendar_date

# Integration Tolerance
rtol = 1e-8 
atol = 1e-9

def evaluationFunction(x, muS, muM, rM, earth, mars):

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
        plots the trajectory and prints the results
    """

    #***********************************************************************************************************************************************************
    # Unpack all design variables
    #***********************************************************************************************************************************************************
    deptDate    = x[0]     # Departure Date
    leg1TOF     = x[1]     # First leg TOF

    arvDate     = deptDate+leg1TOF    # Epoch of Mars Arrival
    
    #***********************************************************************************************************************************************************
    # Leg - 1
    #***********************************************************************************************************************************************************
    earthPosition,earthVelocity = earth.compute_and_differentiate(deptDate)
    marsPosition, marsVelocity = mars.compute_and_differentiate(arvDate)
    earthVelocity = earthVelocity/86400       # [km/s]
    marsVelocity = marsVelocity/86400  # [km/s]
    vEarthDept, vMarsArv, exitflag = LambertIzzo(muS, earthPosition, marsPosition, leg1TOF, int(0))

    # minimize delta-v
    J1 = abs(np.linalg.norm(vEarthDept - vMarsArv))
    Idx = np.argmin(J1)

    #***********************************************************************************************************************************************************
    # Display Results
    #***********************************************************************************************************************************************************

    launchEpoch     = compute_calendar_date(deptDate)
    arrivalEpoch    = compute_calendar_date(arvDate)
    launchDelV      = np.linalg.norm(vEarthDept - earthVelocity)
    arrivalDelV     = np.linalg.norm(vMarsArv - marsVelocity)
    totalDelV       = launchDelV + arrivalDelV

    print('======================== \n')
    print(f'Earth Departure Epoch: {launchEpoch} \n' )
    print(f'Mars Arrival Epoch: {arrivalEpoch} \n')
    print(f'Launch Delta-V: {launchDelV} km/s \n')
    print(f'Arrival Delta-V: {arrivalDelV} km/s \n')
    print(f'Total Delta-V: {totalDelV} km/s \n')
    print('------------------------ \n')

    #***********************************************************************************************************************************************************
    # Plot Trajectory
    #***********************************************************************************************************************************************************
    
    # Propogate Celestial Body Orbits
    statesEarth = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, 365*86400), np.hstack((earthPosition, earthVelocity)), rtol=rtol, atol=atol)
    statesMars  = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, 687*86400), np.hstack((marsPosition, marsVelocity)), rtol=rtol, atol=atol)

    # Propogate Transfer Trajectory
    statesLeg1  = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, leg1TOF*86400), np.hstack((earthPosition, vEarthDept)), rtol=rtol, atol=atol)
    # statesLeg2  = solve_ivp(lambda t, y: two_body_propagator(t, y, muS), (0, leg2TOF*86400), np.hstack((venusPosition, vVenusDept)), rtol=rtol, atol=atol)
    
    # Plot
    fig = plt.figure(figsize=(10, 12)) # width=10 inches, height=6 inches
    ax = fig.add_subplot(111, projection='3d')
    
    # Planet Orbits
    ax.plot(statesEarth.y[0], statesEarth.y[1], statesEarth.y[2], 'b-')
    ax.plot(statesMars.y[0], statesMars.y[1], statesMars.y[2], 'r-')
    
    # Points 
    ax.scatter(earthPosition[0], earthPosition[1], earthPosition[2], color='b', marker='o', label='Earth @ Departure')
    ax.scatter(marsPosition[0], marsPosition[1], marsPosition[2], color='r', marker='o', label='Mars @ Arrival')
    
    # Transfer Trajectory
    ax.plot(statesLeg1.y[0], statesLeg1.y[1], statesLeg1.y[2], 'g-', label='Transfer Trajectory')
    # ax.plot(statesLeg2.y[0], statesLeg2.y[1], statesLeg2.y[2], 'y-', label='Post-FlyBy Trajectory')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Earth-Mars Trajectory')
    ax.legend(loc='upper center', mode='expand', numpoints=1, ncol=4, fancybox = True, fontsize='small')
    ax.grid(True)
    plt.show()
  