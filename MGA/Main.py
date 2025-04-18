# Description 
# This script designs a spacecraft trajectory from Earth to Mars
# The trajectory is designed to optimize delta-V

# Import 
import numpy as np
from jplephem.spk import SPK
from jplephem.calendar import compute_julian_date
from Evaluation import evaluationFunction
from configGA import runGA

# Constants 
muS    = 132712440041.279419     # Gravitational Parameter of Sun [km^2/s^3]
muM    = 0.042828e6              # Gravitational Parameter of Mars [km^2/s^3]
rM     = 3389.5                  # Radius of Mars [km]

# Planet Ephemeris 
ephem = SPK.open("de430.bsp")   # Extract Ephemeris
earth = ephem[0,3]  # Extracts Earth Ephem wrt SolarSystem Barycenter
mars  = ephem[0,4]  # Extracts Mars Ephem wrt SolarSystem Barycenter

# Launch Window 
depLB = compute_julian_date(2020, 1, 1) # Launch window opens
depUB = compute_julian_date(2030, 1, 1) # Launch window closes

# Arrival Window 
minTOF = 100                 # Minimum Duration [days]
maxTOF = 800                 # Maximum Duration [days]

# Optimization 

# Bounds
# [departure date, first leg TOF, Flyby Radius, second leg TOF]
lb = [depLB, minTOF]
ub = [depUB, maxTOF]

# Initial Guess
# initialGuess = np.array([[2460901.16572115, 351.436869240830]])

# Optimize
X_GA, fval_GA, exitflag_GA, output_GA, population, score = runGA(lb, ub, muS, muM, rM, earth, mars)

# Extract Solution 
# X_GA = initialGuess
# evaluationFunction(X_GA, muS, muM, rM, earth, mars)

evaluationFunction(X_GA, muS, muM, rM, earth, mars)