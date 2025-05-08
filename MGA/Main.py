# Description 
# This script designs a spacecraft trajectory from Earth to Mars
# The trajectory is designed to optimize delta-V

# Import 
import numpy as np
import matplotlib.pyplot as plt
from jplephem.spk import SPK
from jplephem.calendar import compute_julian_date, compute_calendar_date
from Evaluation import evaluationFunction
from configGA import runGA
from datetime import datetime

# Constants 
muS    = 132712440041.279419     # Gravitational Parameter of Sun [km^2/s^3]
muM    = 0.042828e6              # Gravitational Parameter of Mars [km^2/s^3]
rM     = 3389.5                  # Radius of Mars [km]

# Planet Ephemeris 
ephem = SPK.open("de430.bsp")   # Extract Ephemeris
earth = ephem[0,3]  # Extracts Earth Ephem wrt SolarSystem Barycenter
mars  = ephem[0,4]  # Extracts Mars Ephem wrt SolarSystem Barycenter

# Launch Window 
depLB = compute_julian_date(2000, 1, 1) # Launch window opens
depUB = compute_julian_date(2030, 1, 1) # Launch window closes

# Arrival Window 
minTOF = 100                 # Minimum Duration [days]
maxTOF = 800                 # Maximum Duration [days]

# Bounds
# [departure date, first leg TOF]
lb = [depLB, minTOF]
ub = [depUB, maxTOF]

# Optional: For testing, you can provide an initial guess
# initialGuess = np.array([[2460901.16572115, 351.436869240830]])

# Run the optimization
print("Starting optimization...")
print(f"Search space: Launch window {compute_calendar_date(depLB)} to {compute_calendar_date(depUB)}")
print(f"Transfer time range: {minTOF} to {maxTOF} days\n")

start_time = datetime.now()
X_GA, fval_GA, exitflag_GA, output_GA, population, score = runGA(lb, ub, muS, muM, rM, earth, mars)
end_time = datetime.now()

print("\nOptimization completed!")
print(f"Time taken: {end_time - start_time}")
print(f"Generations completed: {output_GA['generations_completed']}")
print(f"Exit flag: {exitflag_GA} (0 = max generations reached, 1 = convergence criteria met)")

# Calculate the actual delta-V for the best solution
deptDate = X_GA[0]
leg1TOF = X_GA[1]
arvDate = deptDate + leg1TOF

earthPosition, earthVelocity = earth.compute_and_differentiate(deptDate)
earthVelocity = earthVelocity/86400
marsPosition, marsVelocity = mars.compute_and_differentiate(arvDate)
marsVelocity = marsVelocity/86400

from lambert import LambertIzzo
vEarthDept, vMarsArv, exitflag = LambertIzzo(muS, earthPosition, marsPosition, leg1TOF, 0)

launchDelV = np.linalg.norm(vEarthDept - earthVelocity)
arrivalDelV = np.linalg.norm(vMarsArv - marsVelocity)
totalDelV = launchDelV + arrivalDelV

print("\nBest Solution Found:")
print(f"Departure Date: {compute_calendar_date(deptDate)}")
print(f"Time of Flight: {leg1TOF:.1f} days")
print(f"Arrival Date: {compute_calendar_date(arvDate)}")
print(f"Launch Delta-V: {launchDelV:.2f} km/s")
print(f"Arrival Delta-V: {arrivalDelV:.2f} km/s")
print(f"Total Delta-V: {totalDelV:.2f} km/s")

# Plot optimization history if available
if 'best_solutions_fitness' in output_GA and len(output_GA['best_solutions_fitness']) > 0:
    best_fitness_history = output_GA['best_solutions_fitness']
    
    # Convert fitness to delta-V values (since we're maximizing 1/dV)
    best_dv_history = [1.0/f - 1e-8 for f in best_fitness_history if f > 0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_dv_history)), best_dv_history, 'b-')
    plt.xlabel('Generation')
    plt.ylabel('Best Total ΔV (km/s)')
    plt.title('GA Optimization Progress')
    plt.grid(True)
    plt.savefig('optimization_history.png')
    
    # Plot best solutions over time (if solutions were saved)
    if 'best_solutions' in output_GA:
        best_solutions = output_GA['best_solutions']
        
        # Extract departure dates and TOFs
        departure_dates = [sol[0] if sol is not None else np.nan for sol in best_solutions]
        tofs = [sol[1] if sol is not None else np.nan for sol in best_solutions]
        
        # Convert dates to calendar dates for better readability
        calendar_dates = []
        for date in departure_dates:
            if not np.isnan(date):
                year, month, day = compute_calendar_date(date)
                # Convert floats to integers before formatting
                year_int = int(year)
                month_int = int(month)
                day_int = int(day)
                calendar_dates.append(f"{year_int}-{month_int:02d}-{day_int:02d}")
            else:
                calendar_dates.append(None)
        
        # Create figures for solution evolution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        generations = range(len(best_solutions))
        
        # Plot TOF evolution
        valid_gens = [i for i, tof in enumerate(tofs) if not np.isnan(tof)]
        valid_tofs = [tofs[i] for i in valid_gens]
        
        ax1.plot(valid_gens, valid_tofs, 'r-o')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Time of Flight (days)')
        ax1.set_title('Evolution of Best Time of Flight')
        ax1.grid(True)
        
        # Plot departure date evolution (just show a sample of points for clarity)
        sample_size = min(20, len(valid_gens))
        sample_indices = np.linspace(0, len(valid_gens)-1, sample_size, dtype=int)
        
        departure_dates_plot = [departure_dates[valid_gens[i]] for i in sample_indices]
        calendar_dates_sample = [calendar_dates[valid_gens[i]] for i in sample_indices]
        
        ax2.plot(sample_indices, departure_dates_plot, 'go-')
        ax2.set_xlabel('Sample Point')
        ax2.set_ylabel('Departure Date (Julian Date)')
        ax2.set_title('Evolution of Best Departure Date')
        ax2.grid(True)
        
        # Add date annotations to a few points
        for i, idx in enumerate(sample_indices):
            if i % 4 == 0:  # Annotate every 4th point to avoid clutter
                ax2.annotate(calendar_dates_sample[i], 
                           (idx, departure_dates_plot[i]),
                           textcoords="offset points",
                           xytext=(0,10), 
                           ha='center')
        
        plt.tight_layout()
        plt.savefig('solution_evolution.png')

# Evaluate and plot the trajectory
print("\nPlotting trajectory...")
evaluationFunction(X_GA, muS, muM, rM, earth, mars)

# x_opt = [np.float64(2461357.191841862), np.float64(272.25624876673083)] for 1/1/2020 to 1/1/2030 w/ 100-800 TOF