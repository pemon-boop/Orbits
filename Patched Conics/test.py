# Import required libraries
import os
import sys
import numpy as np
from math import radians, sqrt 
from jplephem.spk import SPK
from jplephem.calendar import compute_julian_date
from lambert import LambertIzzo
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Extract Ephemeris
ephem = SPK.open("de430.bsp")

# Constants
rE = 6378.18        # Earth Radius [km]
rM = 3389.5        # Radius of Mars [km]
muE = 398600        # Grav Parameter of Earth [km3/s2]
muM = 0.042828e6        # Grav Parameter of Mars [km3/s2]
muS = 1.32712e11    # Grav Parameter of Sun [km3/s2]
earth = ephem[0,3]  # Extracts Earth Ephem wrt SolarSystem Barycenter
mars = ephem[0,4]  # Extracts Mars Ephem wrt SolarSystem Barycenter

# Dates
launchDate  = compute_julian_date(2025, 1, 1) # launch Date
arrivalDate = compute_julian_date(2025, 4, 1) # arrival Date
tof = arrivalDate - launchDate  # Time of Flight
nrev = 0

# parking orbit
hParking = 400                         # Altitude of parking orbit around Earth (km)
rParking = rE+hParking
vParking = sqrt(muE/rParking)

# target orbit
hTarget = 400                          # Altitude of target orbit around Mars (km)
rTarget = rM+hTarget
vTarget = sqrt(muM/rTarget)

rEarth, vEarth = earth.compute_and_differentiate(launchDate)
rMars, vMars = mars.compute_and_differentiate(arrivalDate)
vEarth = vEarth/86400       # [km/s]
vMars = vMars/86400       # [km/s]

vSatDept, vSatArv, exitflag = LambertIzzo(muS, np.array(rEarth), np.array(rMars), tof, nrev)