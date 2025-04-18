# Import 
import logging
import numpy as np
import pygad
import time
from Objective import costFunction

def runGA(lb, ub, x0, muS, muV, rV, earth, venus, mars):

    """
    Runs the Genetic Algorithm.
    
    Inputs:
        lb  - Lower bounds (list or numpy array) for design variables.
        ub  - Upper bounds (list or numpy array) for design variables.
        muS, muV, rV - Additional parameters passed to the cost function.
        x0  - Initial population matrix (each row is an individual).
    
    Outputs:
        best_solution  - Best solution found (1D numpy array).
        best_fitness   - Fitness value of the best solution.
        exit_flag      - An integer flag indicating stopping reason (0 if max generations reached).
        output         - A dictionary with information about the optimization process.
        final_population - The final population matrix.
        final_scores   - The scores (fitness values) for the final population.
    """

    #***********************************************************************************************************************************************************
    # Modify the default GA options
    #***********************************************************************************************************************************************************
    
    nvars = len(lb)
    num_gen  = 10
    pop_size = 20 * nvars

    #***********************************************************************************************************************************************************
    # Initial Guess
    #***********************************************************************************************************************************************************
    
    if x0 is None:
        initial_population = np.random.uniform(lb, ub, size=(pop_size, nvars))
    else:
        initial_population = x0
    
    #***********************************************************************************************************************************************************
    # GA options
    #***********************************************************************************************************************************************************
    
    stallGen = "saturate_150"

    #***********************************************************************************************************************************************************
    # Fitness and selection options
    #***********************************************************************************************************************************************************
    eliteCount = 5
    matingPoolSize = int(0.5 * num_gen*nvars)
    parentSelectionType = "rws"             # Roulette Wheel Selection
    crossoverType = "single_point"
    mutationType = "random"
    mutationPercentage = 15 

    #***********************************************************************************************************************************************************
    # Log GA results
    #***********************************************************************************************************************************************************
    
    level = logging.DEBUG
    name = 'logfile.txt'

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(name,'a+','utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    #***********************************************************************************************************************************************************
    # Run the GA
    #***********************************************************************************************************************************************************
    
    # Define the parameter dictionary for passing to the fitness function.
    fitness_function = lambda ga_instance, sol, sol_idx: costFunction(ga_instance, sol, sol_idx, muS, muV, rV, earth, venus, mars)
    
    # Record start time
    start_time = time.time()
    
    # Create the GA instance.
    ga_instance = pygad.GA(num_generations=num_gen,
                           num_parents_mating=matingPoolSize,  # default mating pool size
                           fitness_func=fitness_function,
                           sol_per_pop=pop_size,
                           num_genes=nvars,
                           init_range_low=lb,
                           init_range_high=ub,
                           initial_population=initial_population,
                           parent_selection_type=parentSelectionType,
                           crossover_type=crossoverType,
                           mutation_type=mutationType,
                           mutation_percent_genes=mutationPercentage,  
                           keep_elitism=eliteCount,  
                           stop_criteria=stallGen, # StallGenLimit and MaxTime (6 hours)
                        #    parallel_processing = 5, 
                           logger=logger)  
    
    # Run the GA
    ga_instance.run()
    
    # Record stop time and compute elapsed time.
    elapsed_time = time.time() - start_time
    
    # Retrieve the best solution and its fitness
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    
    # Create an exit flag (0 if maximum generations were reached)
    if ga_instance.generations_completed < num_gen:
        exit_flag = 1  # stopped by criteria other than reaching max generations
    else:
        exit_flag = 0  # reached maximum generations
    
    # Prepare output information similar to MATLAB's output structure.
    output = {
        'generations_completed': ga_instance.generations_completed,
        'time_elapsed': elapsed_time,
        'best_solution_index': best_solution_idx
    }
    
    final_population = ga_instance.population
    final_scores = ga_instance.last_generation_fitness
    
    return best_solution, best_solution_fitness, exit_flag, output, final_population, final_scores
