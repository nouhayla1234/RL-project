from deap import base, creator
 
 
import random
from deap import tools
import numpy as np
import matplotlib.pyplot as plt
import retrieve_plotv4
 
nVar = 3
VarMin=0.3;         # Lower Bound of Variables
VarMax= 1.2;         # Upper Bound of Variables
 
from deap import base, creator
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
 
toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform,VarMin,VarMax)
 
 
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=nVar)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
 
 
"""def rosenbrock(x):
    fit = 0
    for i in range(nVar-1):
        term1 = x[i + 1] - x[i]**2
        term2 = 1 - x[i]
        fit = fit + 100 * term1**2 + term2**2
    return (fit,)"""
 
from functools import reduce
 
def evaluate(speed_ratio):
    pumpsIds = ["PUMP71","PUMP220","PUMP229"]
    dict_pumps = retrieve_plotv4.read_performance_curves('anytown_master.spr')
    P_fuctions = [retrieve_plotv4.polyfit_curves(element, dict_pumps) for element in pumpsIds]
    Efficiencies=[]
    counter = 0
    for element in pumpsIds:
        Q_star = dict_pumps[element]['volume_flow_rate']
        P_star = P_fuctions[counter](Q_star)
        H_star = dict_pumps[element]['headloss']
        Efficiencies.append(retrieve_plotv4.get_pump_efficiencies(Q_star, P_star, H_star))
        counter +=1
    dict_pumps = retrieve_plotv4.apply_affinity_laws(pumpsIds,dict_pumps,speed_ratio)
    for element in pumpsIds:
        retrieve_plotv4.write_performance_curve(element, 'new_anytown_master.spr', dict_pumps)
    retrieve_plotv4.launch_staci('new_anytown_master.spr')
     
    return (reduce(lambda x, y: x*y,Efficiencies),)
 
 
"""def evaluate(individual):
    return rosenbrock(individual)"""
 
 
 
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes , indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evaluate)
 
 
# numdim
 
 
 
def main():
    pop = toolbox.population(n=2)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 5
 
     
    list_max_scores = []
    indiv_max = []
    # Evaluate the entire population
     
    fitnesses = list(map(toolbox.evaluate, pop))
     
 
 
    for ind, fit in zip(pop, fitnesses):
         
        ind.fitness.values = fit
         
 
    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
 
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
 
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
 
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
 
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        fitnesses = list(map(toolbox.evaluate, pop))
        fitnesses_list = []
        for element in fitnesses:
            fitnesses_list.append(element[0])
         
        list_max_scores.append(max(fitnesses_list))
        indiv_max.append(pop[fitnesses_list.index(max(fitnesses_list))])
    print(indiv_max)
    plt.plot(list_max_scores, 'r', label='Best score of a generation')
     
    plt.ylabel('Rosenbrock')
    plt.xlabel('Generation')
    plt.show()
     
     
    return pop
 
 
 
 
 
 
 
 
if __name__ == "__main__":
    main()