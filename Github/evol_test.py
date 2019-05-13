'''
EVOL_TEST


Change the demands of original_file for each row of the data_demand.csv 

Then compute genetic algorithm on action_file

And write the optimal reward and speed_ratio in the corresponding row of the data_demand.csv 


'''


from deap import base, creator
from functools import reduce
from deap import base, creator
import random
from deap import tools
import numpy as np
import matplotlib.pyplot as plt
import retrieve_plotv6 as ret
from tqdm import tqdm
import csv
import xml.etree.ElementTree as ET
 

nVar = 3
VarMin=0.3;         # Lower Bound of Variables
VarMax= 1.2;         # Upper Bound of Variables
 

#INIT
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform,VarMin,VarMax)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=nVar)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



original_file = 'anytown_master_test.spr'
action_file = 'anytown_master_test_random.spr'
filepath = 'data_demand.csv'



def retrieve_csvdata():
    with open(filepath, 'r') as readFile:
        reader = csv.reader(readFile)
        csvData = list(reader)
    readFile.close()
    return(csvData)


def write_csvdata(csvData):
    with open(filepath, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def randomize_demand_csv(active_row, csvData): # We change the demand of the original_file
    tree = ET.parse(original_file)
    root = tree.getroot()
    counter = 2 # 0:reward, 1:speed ratio, 2~31:random demand

    for child in root:
        for node in child.findall('node'):
            demand = node.find('demand')
            new_demand = csvData[active_row][counter]
            demand.text = new_demand
            counter += 1

    tree.write(original_file, encoding='utf8')
 


def evaluate(speed_ratio, dict_pumps_original):
    pumpsIds = ["PUMP71","PUMP220","PUMP229"]
    dict_pumps_original = ret.read_performance_curves(original_file)
    dict_pumps = ret.apply_affinity_laws(pumpsIds,dict_pumps_original,speed_ratio)
    for element in pumpsIds:
        ret.write_performance_curve(element, original_file, action_file, dict_pumps)
    ret.launch_staci(action_file)
    dict_pumps = ret.read_performance_curves(action_file)
    P_fuctions = [ret.polyfit_curves(element, dict_pumps) for element in pumpsIds]
    Efficiencies=[]
    counter = 0
    for element in pumpsIds:
        Q_star = dict_pumps[element]['volume_flow_rate']
        P_star = P_fuctions[counter](Q_star)
        H_star = dict_pumps[element]['headloss']
        Efficiencies.append(ret.get_pump_efficiencies(Q_star, P_star, H_star))
        counter +=1
    
    return (reduce(lambda x, y: x*y,Efficiencies),)


 
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes , indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("evaluate", evaluate)
 


 
 
def main():

    '''INIT'''
    csvData = retrieve_csvdata() # Add the data to csvData
    active_row = 0 # Begin with row 0 of csvData

    number_of_tests = len(csvData) # Number of rows from csvData
    print("Number of tests : ", number_of_tests)

    pbar = tqdm([i for i in range(number_of_tests)])

    for h in range(number_of_tests):

        randomize_demand_csv(active_row, csvData) # Test on the active row (one set of demands)
        dict_pumps_original = ret.read_performance_curves(original_file)

        pbar.update()
        pbar.set_description("Computing Test number %i" % (h+1))

        pop = toolbox.population(n=100) ##########
        CXPB, MUTPB, NGEN = 0.5, 0.1, 150 ###############
        pbar2 = tqdm([i for i in range(NGEN)])

        list_max_scores = []
        indiv_max = []
        dict_score_pop = {}
        # Evaluate the entire population
        
        fitnesses = list(map(toolbox.evaluate, pop, dict_pumps_original))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
     
        for g in range(NGEN):
            pbar2.update()
            pbar2.set_description("Computing generation %i of test %i" % (g+1, h+1))
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
            fitnesses = map(toolbox.evaluate, invalid_ind, dict_pumps_original)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
     
            # The population is entirely replaced by the offspring
            pop[:] = offspring
            fitnesses = list(map(toolbox.evaluate, pop, dict_pumps_original))
            fitnesses_list = []
            for element in fitnesses:
                fitnesses_list.append(element[0])
             
            list_max_scores.append(max(fitnesses_list))
            indiv_max.append(pop[fitnesses_list.index(max(fitnesses_list))])
            dict_score_pop[max(fitnesses_list)] = pop[fitnesses_list.index(max(fitnesses_list))]


        max_score = max(list_max_scores)
        best_pop = dict_score_pop[max_score]
        best_score = max_score

        #print('Best pop : ', best_pop)
        #print('max score : ', max_score)


        csvData[active_row][0] = str(best_score) #Add the reward
        csvData[active_row][1] = str(best_pop) #Add the speed ratio
        active_row += 1
        pbar2.close()

    print("\nComputing done\nWriting to new csvData to '%s'" % (filepath))
    write_csvdata(csvData)
    print("Writing done.")




if __name__ == "__main__":
    main()