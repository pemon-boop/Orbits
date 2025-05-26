# Import 
import logging
import numpy as np
import pygad
import time
from Objective import costFunction
from jplephem.calendar import compute_julian_date

def custom_mutation(offspring, ga_instance):
    """
    Custom mutation function that applies different mutation strategies
    for departure date and TOF parameters.
    
    Parameters:
        offspring - The offspring to be mutated
        ga_instance - The GA instance
        
    Returns:
        The mutated offspring
    """
    # Get mutation probability per gene (mutation percent genes from GA instance)
    mutation_probability = ga_instance.mutation_percent_genes
    
    # If not specified, use a default value (e.g., 10%)
    if mutation_probability is None:
        mutation_probability = 0.1
    
    # PyGAD may call this with a single solution or multiple solutions
    # Check if we're dealing with a single solution or multiple solutions
    if offspring.ndim == 1:
        # Single solution case
        num_genes = len(offspring)
        
        # Mutation based on gene type
        for gene_idx in range(num_genes):
            # Random value to determine whether to apply mutation
            random_value = np.random.random()
            
            if random_value <= mutation_probability:
                # Get gene bounds
                lb = ga_instance.gene_space[gene_idx][0]
                ub = ga_instance.gene_space[gene_idx][1]
                range_width = ub - lb
                
                # Different mutation strategy based on gene index
                if gene_idx == 0:  # Departure date
                    # Allow larger jumps in the date range with 10% probability
                    if np.random.random() < 0.1:
                        # Major jump to different region of the search space
                        offspring[gene_idx] = np.random.uniform(lb, ub)
                    else:
                        # For dates, we might want smaller mutations (e.g., +/- 30 days)
                        mutation_range = min(float(range_width * 0.05), 30.0)  # 5% of range or 30 days, whichever is smaller
                        mutation = np.random.uniform(-mutation_range, mutation_range)
                        offspring[gene_idx] += mutation
                    
                elif gene_idx == 1:  # TOF
                    # For TOF, allow larger variations (e.g., +/- 10% of current value)
                    gene_value = float(offspring[gene_idx])
                    range_width_value = float(range_width)
                    mutation_amplitude = max(gene_value * 0.1, range_width_value * 0.05)
                    mutation = np.random.uniform(-mutation_amplitude, mutation_amplitude)
                    offspring[gene_idx] += mutation
                    
                # Ensure the gene stays within bounds
                if offspring[gene_idx] < lb:
                    offspring[gene_idx] = lb
                elif offspring[gene_idx] > ub:
                    offspring[gene_idx] = ub
    else:
        # Multiple solutions case - iterate through each solution
        for sol_idx in range(offspring.shape[0]):
            solution = offspring[sol_idx, :]
            
            # Number of genes in each solution
            num_genes = len(solution)
            
            # Mutation based on gene type
            for gene_idx in range(num_genes):
                # Random value to determine whether to apply mutation
                random_value = np.random.random()
                
                if random_value <= mutation_probability:
                    # Get gene bounds
                    lb = ga_instance.gene_space[gene_idx][0]
                    ub = ga_instance.gene_space[gene_idx][1]
                    range_width = ub - lb
                    
                    # Different mutation strategy based on gene index
                    if gene_idx == 0:  # Departure date
                        # Allow larger jumps in the date range with 10% probability
                        if np.random.random() < 0.1:
                            # Major jump to different region of the search space
                            solution[gene_idx] = np.random.uniform(lb, ub)
                        else:
                            # For dates, we might want smaller mutations (e.g., +/- 30 days)
                            mutation_range = min(float(range_width * 0.05), 30.0)
                            mutation = np.random.uniform(-mutation_range, mutation_range)
                            solution[gene_idx] += mutation
                        
                    elif gene_idx == 1:  # TOF
                        # For TOF, allow larger variations (e.g., +/- 10% of current value)
                        gene_value = float(solution[gene_idx])
                        range_width_value = float(range_width)
                        mutation_amplitude = max(gene_value * 0.1, range_width_value * 0.05)
                        mutation = np.random.uniform(-mutation_amplitude, mutation_amplitude)
                        solution[gene_idx] += mutation
                        
                    # Ensure the gene stays within bounds
                    if solution[gene_idx] < lb:
                        solution[gene_idx] = lb
                    elif solution[gene_idx] > ub:
                        solution[gene_idx] = ub
                    
                    # Update the offspring array
                    offspring[sol_idx, gene_idx] = solution[gene_idx]
                
    return offspring

def generate_initial_population(pop_size, lb, ub, earth, mars, muS):
    """
    Generate a diverse initial population for the Earth-Mars trajectory problem.
    
    Parameters:
        pop_size - Size of the population
        lb - Lower bounds for design variables
        ub - Upper bounds for design variables
        earth - Earth ephemeris
        mars - Mars ephemeris
        muS - Sun's gravitational parameter
        
    Returns:
        initial_population - Array of shape (pop_size, 2) with candidates
    """
    # Initialize population array
    initial_population = np.zeros((pop_size, len(lb)))
    
    # Get the total date range
    date_range = ub[0] - lb[0]
    
    # Divide the date range into segments to ensure coverage
    num_segments = 20
    segment_size = date_range / num_segments
    
    # Distribute population across segments to ensure full coverage
    for i in range(pop_size):
        # Choose a segment for this individual (ensures even distribution)
        segment = i % num_segments
        
        # Generate a date within this segment
        segment_start = lb[0] + segment * segment_size
        segment_end = segment_start + segment_size
        
        # Set departure date within the segment
        initial_population[i, 0] = np.random.uniform(segment_start, segment_end)
        
        # Set time of flight with different strategies
        if i % 3 == 0:
            # Short transfers (near minimum)
            initial_population[i, 1] = np.random.uniform(lb[1], lb[1] + 100)
        elif i % 3 == 1:
            # Medium transfers (Hohmann-like)
            initial_population[i, 1] = np.random.uniform(200, 350)
        else:
            # Long transfers
            initial_population[i, 1] = np.random.uniform(350, ub[1])
    
    # Also include some focused solutions around known launch windows (20% of population)
    n_window_candidates = int(pop_size * 0.2)
    
    # Historical good Earth-Mars transfer windows (around opposition dates)
    historical_windows = [
        compute_julian_date(2014, 1, 1),
        compute_julian_date(2016, 3, 1),
        compute_julian_date(2018, 5, 1),
        compute_julian_date(2020, 7, 1),   # 2020 window
        compute_julian_date(2022, 9, 1),   # 2022 window
        compute_julian_date(2024, 11, 1),  # 2024 window
        compute_julian_date(2027, 1, 1),   # 2027 window
        compute_julian_date(2029, 3, 1),   # 2029 window
    ]
    
    # Replace some solutions with ones near historical windows
    for i in range(n_window_candidates):
        idx = pop_size - n_window_candidates + i  # Start replacing from the end
        window = np.random.choice(historical_windows)
        
        # Add some randomness around the window date (+/- 60 days)
        initial_population[idx, 0] = window + np.random.uniform(-60, 60)
        
        # TOF around typical Hohmann transfer time (~260 days) with variations
        initial_population[idx, 1] = 260 + np.random.uniform(-60, 140)
    
    return initial_population

def runGA(lb, ub, muS, muM, rM, earth, mars, x0=None):
    """
    Runs the Genetic Algorithm.
    
    Inputs:
        lb  - Lower bounds (list or numpy array) for design variables.
        ub  - Upper bounds (list or numpy array) for design variables.
        muS, muM, rM - Additional parameters passed to the cost function.
        earth, mars - Ephemeris objects for the planets
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
    num_gen  = 1000
    pop_size = 500 * nvars

    #***********************************************************************************************************************************************************
    # Initial Population
    #***********************************************************************************************************************************************************

    if x0 is None:
        initial_population = generate_initial_population(pop_size, lb, ub, earth, mars, muS)
    else:
        initial_population = x0

    #***********************************************************************************************************************************************************
    # GA options
    #***********************************************************************************************************************************************************
    
    # Define gene space (bounds for each variable)
    gene_space = []
    for i in range(nvars):
        gene_space.append([lb[i], ub[i]])
    
    # Early stopping criteria
    stallGen = "saturate_150"  # Stop if no improvement for 150 generations

    #***********************************************************************************************************************************************************
    # Fitness and selection options
    #***********************************************************************************************************************************************************
    eliteCount = 5  # Reduced to prevent premature convergence
    matingPoolSize = int(0.5 * pop_size)  # Using half of population for mating
    
    # Selection method options
    parentSelectionType = "rws"  # Rank selection for reduced selection pressure
    
    # Use two-point crossover to better preserve date ranges
    crossoverType = "two_points"
    crossoverProbability = 0.8
    
    # Use our custom mutation with improved exploration
    mutationType = custom_mutation
    mutationPercentage = 0.2  # Increased mutation rate for better exploration

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
    # Custom callback to monitor diversity
    #***********************************************************************************************************************************************************
    def on_generation(ga_instance):
        if ga_instance.generations_completed % 10 == 0:
            # Calculate diversity statistics for departure dates
            departure_dates = ga_instance.population[:, 0]
            min_date = np.min(departure_dates)
            max_date = np.max(departure_dates)
            std_date = np.std(departure_dates)
            
            # Log diversity information
            print(f"Generation {ga_instance.generations_completed}: Date range = [{min_date:.1f}, {max_date:.1f}], Std = {std_date:.1f}")
            
            # If diversity is getting too low, introduce some random solutions
            if std_date < (ub[0] - lb[0]) / 50:  # If std dev is less than 2% of the range
                print("Low diversity detected - introducing new random solutions")
                worst_idx = np.argsort(ga_instance.last_generation_fitness)[:pop_size//10]  # Replace 10% worst solutions
                for idx in worst_idx:
                    ga_instance.population[idx, 0] = np.random.uniform(lb[0], ub[0])
                    ga_instance.population[idx, 1] = np.random.uniform(lb[1], ub[1])
        return False
    
    #***********************************************************************************************************************************************************
    # Run the GA
    #***********************************************************************************************************************************************************
    
    # Define the parameter dictionary for passing to the fitness function.
    fitness_function = lambda ga_instance, sol, sol_idx: costFunction(ga_instance, sol, sol_idx, muS, muM, rM, earth, mars)
    
    # Record start time
    start_time = time.time()
    
    # Create the GA instance.
    ga_instance = pygad.GA(num_generations=num_gen,
                           num_parents_mating=matingPoolSize,
                           fitness_func=fitness_function,
                           sol_per_pop=pop_size,
                           num_genes=nvars,
                           gene_space=gene_space,  # Using gene_space instead of init_range
                           initial_population=initial_population,
                           parent_selection_type=parentSelectionType,
                           crossover_type=crossoverType,
                           crossover_probability=crossoverProbability,
                           mutation_type=mutationType,
                           mutation_percent_genes=mutationPercentage,  
                           keep_elitism=eliteCount,  
                           stop_criteria=stallGen,
                           save_best_solutions=True,  # Save best solutions for analysis
                           on_generation=on_generation,  # Add diversity monitoring
                           parallel_processing=None,  # Disable parallel processing to avoid issues
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
        'best_solution_index': best_solution_idx,
        'best_solutions_fitness': ga_instance.best_solutions_fitness,  # Fitness history
        'best_solutions': ga_instance.best_solutions                  # Solution history
    }
    
    final_population = ga_instance.population
    final_scores = ga_instance.last_generation_fitness
    
    return best_solution, best_solution_fitness, exit_flag, output, final_population, final_scores