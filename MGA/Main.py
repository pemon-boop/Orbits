# Description 
# This script designs a spacecraft trajectory from Earth to Mars with a Venus flyby. 
# The trajectory is divided into three phases:
#    1. First Leg : Earth to Venus
#    2. Gravity Assist at Venus
#    3. Second Leg : Venus to Mars
# The trajectory is designed to only achieve a Venus flyby and not optimize
# any mission parameters. 

# Import 
import numpy as np
from jplephem.spk import SPK
from jplephem.calendar import compute_julian_date
from Evaluation import evaluationFunction
from configGA import runGA

# Constants 
muS    = 132712440041.279419     # Gravitational Parameter of Sun [km^2/s^3]
muV    = 324860                  # Gravitational Parameter of Venus [km^2/s^3]
rV     = 6051.8                  # Radius of Venus [km]

# Planet Ephemeris 
ephem = SPK.open("de430.bsp")   # Extract Ephemeris
earth = ephem[0,3]  # Extracts Earth Ephem wrt SolarSystem Barycenter
venus = ephem[0,2]  # Extracts Venus Ephem wrt SolarSystem Barycenter
mars  = ephem[0,4]  # Extracts Mars Ephem wrt SolarSystem Barycenter

# Launch Window 
depLB = compute_julian_date(2025, 1, 1) # Launch window opens
depUB = compute_julian_date(2027, 1, 1) # Launch window closes

# Flyby Window 
leg1minTOF = 30                  # Minimim Leg-1 Duration [days]
leg1maxTOF = 400                 # Maximum Leg-1 Duration [days]

# Arrival Window 
leg2minTOF = 100                 # Minimim Leg-2 Duration [days]
leg2maxTOF = 400                 # Maximum Leg-2 Duration [days]

# Flyby Constraint 
minFlyByRadius = 1.05            # Minimum Flyby Radius (1.05*Rp)
maxFlybyRadius = 5               # Maximum Flyby Radius (5*Rp)

# Optimization 

# Bounds
# [departure date, first leg TOF, Flyby Radius, second leg TOF]
lb = [depLB, leg1minTOF, minFlyByRadius, leg2minTOF]
ub = [depUB, leg1maxTOF, maxFlybyRadius, leg2maxTOF]

# Initial Guess
initialGuess = np.array([[2460901.16572115], [56.0778218004728], [2.44606067026515], [351.436869240830]])

# Optimize
X_GA, fval_GA, exitflag_GA, output_GA, population, score = runGA(lb, ub, initialGuess, muS, muV, rV, earth, venus, mars)

# Extract Solution 
# X_GA = initialGuess
# evaluationFunction(X_GA, muS, muV, rV, earth, venus, mars)
