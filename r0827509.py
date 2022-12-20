import Reporter
import numpy as np
import pandas as pd
import random
import math
import time
from numba import jit
import warnings
warnings.filterwarnings('ignore')

class Parameters:
    '''
    Abstract to store all modifiable parameters of the genetic algorithm.
    Params:
        lamda (int) : population size
        mu (int) : number of offsprings to generate per island per iteration
        k (int) : k value in k tournament selection
        standard_alfa (float) : mutation probability
        random_share (float) : share of the population initialized randomly
        elimination (str) : elimination strategy. 'lambda+mu' or 'lambda+musharing'
        n_LSO (int) : number of individuals to improve with local search per iteration
        n_swap (int) : number of individuals to swap between islands during the island swap rounds

    '''
    def __init__(self, lamda=50, mu=125, k=15, its=100, standard_alfa=0.3, random_share=0.8,elimination='lambda+musharing',n_LSO=20,n_swap=5):
        self.lamda = lamda
        self.its = its
        self.mu = mu
        self.k = k
        self.standard_alfa = standard_alfa
        self.random_share = random_share
        self.elimination = elimination
        self.n_LSO = n_LSO
        self.n_swap = n_swap

class r0827509:
    '''
    Main class to run the genetic algorithm.
    Params :
        reporter (Reporter) : reporter object to evaluate the remaining time and store fitness values in a csv file
    '''

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        
    def optimize(self, 
                 filename,
                 parameters = Parameters(),
                 to_csv=True):
        '''
        Execute the genetic algorithm on a given tsp problem.
        Params : 
            filename (str): name of the file containing the tsp distance matrix
            parameters (Parameters) : the parameters of the algorithm
        '''
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        tsp = TravellingSalesmanProblem(distanceMatrix)
        yourConvergenceTestsHere = True

        #Graph plotting data
        df_data = []
        type = []
        iteration = []

        while( yourConvergenceTestsHere):
            ##################
            # Initialization #
            ##################
            
            start_time = time.time()
            random_pop, greedy_pop = heur_initialize(tsp,parameters)
            population = greedy_pop[:len(greedy_pop)//2] + random_pop[:len(random_pop)//2] + greedy_pop[len(greedy_pop)//2:] + random_pop[:len(random_pop)//2]
            #Perform local search on the initial population
            for idx in range(0,min(parameters.n_LSO,len(population))):
                LSO(population[idx])
            # Evaluate fitness and report
            fitnesses = [indiv.fitness for indiv in population]
            bestObjective = max(fitnesses)
            meanObjective=  mean(fitnesses)
            df_data += [-bestObjective,-meanObjective]
            type += ['best','mean']
            iteration+= [0,0]
            print(0, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
            print('Time needed to initialize the population : %s seconds'%(time.time()-start_time))

            ##################
            # Create islands #
            ##################
            half_pop = round(len(population)/2)
            island1 = population[:half_pop]
            island2 = population[half_pop:]

            #############
            # Main loop #
            #############
            same_best_counter = 0
            for i in range(1, parameters.its+1): 
                #Control iteration execution time
                iter_start_time = time.time()
                #Every 3 iterations, perform a swap of individuals between the islands
                if i%3==0:
                    island1, island2 = self.swap_islands(island1,island2,parameters)
                #Update the islands population
                island1 = self.create_new_generation(tsp,island1,parameters,[swap_mutate,insertion_mutate], OrderCrossover)
                island2 = self.create_new_generation(tsp,island2,parameters,[swap_mutate,insertion_mutate], CSOX)
                population = island1 + island2
                # Evaluate fitness and reporting
                fitnesses = [indiv.fitness for indiv in population]
                #Report whether the best objective has changed or not
                if bestObjective == max(fitnesses):
                    same_best_counter +=1
                else:
                    same_best_counter=0
                #Report results
                bestObjective = max(fitnesses)
                bestSolution = population[fitnesses.index(bestObjective)]
                meanObjective = mean(fitnesses)
                df_data += [-bestObjective,-meanObjective]
                type += ['best','mean']
                iteration += [i,i]
                #Print intermediate results
                print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
                #Evaluate the time remaining
                timeLeft = self.reporter.report(meanObjective, bestObjective, np.array(bestSolution.order))
                iter_duration = time.time()-iter_start_time
                if timeLeft < iter_duration * 1.1:
                    #The remaining time is lower than the execution time of the last iteration (+a 10% margin)
                    print('No time left')
                    print('Elapsed time : %s seconds'%(time.time()-start_time))
                    print('Current iter %s'%i)
                    print('Results after timeout')
                    print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
                    finalFitness = bestObjective
                    finalMeanFitness = meanObjective
                    break
                if same_best_counter>=50:
                    #Not enough improvements after 50 iterations
                    print('Same best for 50 iterations')
                    print('Elapsed time : %s seconds'%(time.time()-start_time))
                    finalFitness = bestObjective
                    finalMeanFitness = meanObjective
                    print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
                    
                    break
                if i == parameters.its:
                    #Maximum number of iterations reached
                    print('Optimization finished')
                    print('Elapsed time : %s seconds'%(time.time()-start_time))
                    print('Results after all iterations')
                    print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)  
                    finalFitness = bestObjective
                    finalMeanFitness = meanObjective
                    break
            
            yourConvergenceTestsHere = False

            if to_csv:
                results_df = pd.DataFrame({'iteration':iteration,'fitness':df_data,
                                           'type':type})
                results_df.to_csv('results_'+filename,index=False)
        return finalFitness, finalMeanFitness

    
    def create_new_generation(self,tsp,population,parameters,mutation,recombination,elitism = True):
        """
        Updates a population using recombination, mutation, local search, and elimination operators.
        Params:
            tsp (TravelingSalesmanProblem) : a tsp problem
            population (list[CandidateSolutions]) : a list of individuals to update
            parameters (Parameters) : the parameters of the algorithm
            mutation (list) : a list of mutation functions to apply sequentially on the individuals
            recombination (func) : a recombination operator to generate offsprings
            elitism (bool) : if True, apply elitism to preserve the current best value in the population
        Returns : 
            population (list[CandidateSolutions]) : the updated list of individuals
        """

        elimination_dict = {'lambda+mu':elimination,
                            'lambda+musharing':elimination_fitness_sharing,
                            }
        elimination_op = elimination_dict[parameters.elimination]
        offspring = list()
        for _ in range(1, parameters.mu):   
            #Select parents and generate offsprings
            p_1 = selection(population, parameters.k)
            p_2 = selection(population, parameters.k)
            for order in recombination(p_1.order,p_2.order):
                child = CandidateSolution(tsp, order=order)
                for mut_op in mutation:
                    #Apply mutations on the offsprings
                    child = mut_op(child,parameters)
                #Compute the fitness of the offsprings
                child.computeFitness()
                offspring.append(child)
        if elitism:
            #Save the best individual and remove it from the population  #Should still be used to generate the offsprings first
            elite = elimination(population,[],1) 
            population.remove(elite[0])
            # print(elite[0].fitness)
        else:
            elite = []
        mutated_population = list()
        for ind in population:
            for mut_op in mutation:
                #Mutation of the seed population
                ind = mut_op(ind,parameters)
            ind.computeFitness()
            mutated_population.append(ind)
        population = mutated_population
        #Perform elimination
        population = [elite[0]] + elimination_op(population,offspring, len(population))
        # print(population[0].fitness)
        #Local search on best individuals
        for idx in range(0,min(parameters.n_LSO,len(population))):
            LSO(population[idx])
        return population

    def swap_islands(self,island1,island2,parameters):
        #Swap individuals between two islands
        moveto1 = [selection(island2,parameters.k) for _ in range(parameters.n_swap)]
        moveto2 = [selection(island1,parameters.k) for _ in range(parameters.n_swap)]
        island1 = [i for i in island1 if i not in  moveto2]
        island1 += moveto1
        island2 = [i for i in island2 if i not in  moveto1]
        island2 += moveto2
        return island1, island2


class TravellingSalesmanProblem:
    '''
    A traveling salesman problem.
    Params:
        number_of_cities (int) : the number of cities in the tsp
        valid_transition (dict) : takes as key an origin city and as values a list of all the valid destination cities
        distance_matrix (dict) : nested dictionary containing the distances between cities
    '''
    def __init__(self, distance_matrix):
        self.number_of_cities = len(distance_matrix)
        distance_dict = dict()
        j = 0
        self.valid_transition = {} 
        while j < len(distance_matrix):
            distance_dict[j] = {index: distance_matrix[j][index] for index in range(len(distance_matrix[j]))}
            self.valid_transition[j] = {index for index in range(len(distance_matrix[j])) if not math.isinf(distance_matrix[j][index]) and distance_matrix[j][index]}
            j += 1 
        self.distance_matrix = distance_dict


class CandidateSolution:
    '''
    An individual representing a possible solution for the travelling salesman problem.
    Params:
        tsp (TravellingSalesmanProblem) : an instance of a tsp problem
        age (int) : the number of iterations for which the individual has been in the population. 
                    An individual older than 5 is removed by elimination operators
        order (numpy.array) : the representation of the solution as a sequence of cities
        alfa (float) : the mutation probability
    '''
    @jit
    def __init__(self, tsp, order=None):
        self.age=0
        self.tsp = tsp
        self.shared_fitness = 0
        if order is None:
            order = list(range(0,self.tsp.number_of_cities))
            random.shuffle(order)
            self.order = np.array(order)
        else:
            self.order = order
    
    def computeFitness(self):
        distance = 0
        penalty = 0
        for i in range(len(self.order)-1):
            new_dist = self.tsp.distance_matrix[self.order[i]][self.order[i+1]]
            if math.isinf(new_dist):
                penalty += 1
            distance += new_dist
        last_dist = self.tsp.distance_matrix[self.order[-1]][self.order[0]]
        if math.isinf(last_dist):
            #Add penalities for every infinite cost edge
            penalty +=1
        distance += last_dist   
        if math.isinf(distance):
            #Convert infinite to a very large negative number 
            distance =  1e10
        self.fitness = -distance  - (penalty * 1e3)

@jit
def heur_initialize(tsp, parameters):
    #Heuristic initialization of the population
    random_pop = list()
    greedy_pop = list()
    n_random = round(parameters.lamda * parameters.random_share)
    n_greedy = parameters.lamda - n_random
    count = 0
    while count < n_random:
        #Standard initialization
        indiv = CandidateSolution(tsp)
        count +=1
        indiv.computeFitness()
        random_pop.append(indiv)   
    cities = set(range(0,tsp.number_of_cities))
    taboo = set()
    for _ in range(min(250,tsp.number_of_cities)):
        #Up to 250 greedy initialization, each time with a different starting position
        current_city = random.choice(list(cities - taboo))
        taboo.add(current_city)
        order = [current_city]
        timer = time.time()
        #A maximum of two seconds is allowed to find an individual
        while len(order) < tsp.number_of_cities-1 and time.time() - timer < 2:
            #If no remaining alternatives, backtrack 10 cities earlier and pick a random city among the available options
            if len(set(tsp.valid_transition[current_city])-set(order)) == 0:
                order = order[:-10] 
                current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order])
                order.append(current_city)
            else:
            #Append the cheapest valid transition to the order
                current_city = min(set(tsp.valid_transition[order[-1]]) - set(order), key=tsp.distance_matrix[current_city].__getitem__)
                order.append(current_city)
        if not time.time() - timer > 2:
            remaining_city = [number for number in range(0,tsp.number_of_cities) if number not in order]
            order.append(remaining_city[0])
            order = np.array(order)
            indiv = CandidateSolution(tsp, order)
            indiv.computeFitness()
            greedy_pop.append(indiv)
    
    fitnesses = [indiv.fitness for indiv in greedy_pop]
    sorted_fitnesses = sorted(fitnesses,reverse=True)
    #Keep the best greedy individuals 
    greedy_pop =  [greedy_pop[fitnesses.index(f)]  for f in sorted_fitnesses[:n_greedy]]        
    return random_pop, greedy_pop


def mean(list):
    return sum(list)/len(list)

@jit
def LSO(indiv):
    # 2-opt local search operator
    start_time = time.time()
    #First case : first city is at index 0
    subpath_cost = 0
    inverted_subpath_cost=0
    for j in range(1,len(indiv.order)-1):
        subpath=indiv.order[0:j+1]
        subpath_cost += indiv.tsp.distance_matrix[subpath[-2]][subpath[-1]]
        inverted_subpath = subpath[::-1]
        inverted_subpath_cost += indiv.tsp.distance_matrix[inverted_subpath[0]][inverted_subpath[1]]
        if not math.isinf(inverted_subpath_cost):
            old_cost =  indiv.tsp.distance_matrix[indiv.order[-1]][indiv.order[0]] + indiv.tsp.distance_matrix[indiv.order[j]][indiv.order[j+1]]  + subpath_cost
            new_cost =  indiv.tsp.distance_matrix[indiv.order[0]][indiv.order[j+1]]  + indiv.tsp.distance_matrix[indiv.order[-1]][indiv.order[j]] + inverted_subpath_cost
            if new_cost < old_cost and not math.isinf(new_cost):
                indiv.order[0:j+1] = inverted_subpath
        else:
            #If the inverted subpath has infinite cost, all extensions of it will have infinite cost too and there is no need to evaluate them
            break

    for i in range(1,len(indiv.order)-2):
        #Standard case
        subpath_cost = 0
        inverted_subpath_cost=0
        if time.time() -start_time > 10:
          break
        for j in range(i+1,len(indiv.order)-1):
            subpath=indiv.order[i:j+1]
            subpath_cost += indiv.tsp.distance_matrix[subpath[-2]][subpath[-1]]
            inverted_subpath =subpath[::-1]
            inverted_subpath_cost += indiv.tsp.distance_matrix[inverted_subpath[0]][inverted_subpath[1]]
            if not math.isinf(inverted_subpath_cost):
                old_cost =  indiv.tsp.distance_matrix[indiv.order[i-1]][indiv.order[i]] + indiv.tsp.distance_matrix[indiv.order[j]][indiv.order[j+1]]  + subpath_cost
                new_cost =  indiv.tsp.distance_matrix[indiv.order[i]][indiv.order[j+1]]  + indiv.tsp.distance_matrix[indiv.order[i-1]][indiv.order[j]] + inverted_subpath_cost
                if new_cost < old_cost and not math.isinf(new_cost):
                    indiv.order[i:j+1] = inverted_subpath
            else:
                break
    
    subpath_cost=0
    inverted_subpath_cost=0
    for j in range(0,len(indiv.order)-2):
        subpath = np.concatenate((indiv.order[-1:],indiv.order[0:j+1]))
        subpath_cost += indiv.tsp.distance_matrix[subpath[-2]][subpath[-1]]
        inverted_subpath =subpath[::-1]
        inverted_subpath_cost += indiv.tsp.distance_matrix[inverted_subpath[0]][inverted_subpath[1]]
        if not math.isinf(inverted_subpath_cost):
            old_cost =  indiv.tsp.distance_matrix[indiv.order[-2]][indiv.order[-1]] + indiv.tsp.distance_matrix[indiv.order[j]][indiv.order[j+1]]  + subpath_cost
            new_cost =  indiv.tsp.distance_matrix[indiv.order[-1]][indiv.order[j+1]]  + indiv.tsp.distance_matrix[indiv.order[-2]][indiv.order[j]] + inverted_subpath_cost
            if new_cost < old_cost and not math.isinf(new_cost):
                indiv.order[-1] = inverted_subpath[0]
                indiv.order[0:j+1] = inverted_subpath[1:]
        else:
            break   
    indiv.computeFitness()



def selection(population, k):
    # k-tournament selection 
    selected = random.sample(population, k)
    fitnesses = [indiv.fitness for indiv in selected] 
    return selected[fitnesses.index(max(fitnesses))]


def sort_according_to_fitnesses(candidate_solutions_list):
    #Sort individuals by decreasing order based on their fitness
    return sorted(candidate_solutions_list, key=lambda sol: sol.fitness, reverse=True)


def hamming_distance(ind1,new_generation):
    #Compute the hamming distance between one offspring and the new generation of individuals
    dists = []
    for ind2 in new_generation:
        town_start = ind1[0]
        idx_start_2 = np.where(ind2.order==town_start)[0][0]
        sorted_ind2 = np.concatenate((ind2.order[idx_start_2:],ind2.order[:idx_start_2]))
        dist = 0
        for i in range(len(ind1)):
            if ind1[i] != sorted_ind2[i]:
                dist+=1
        dists.append(dist)
    return dists
    
    
def sort_according_to_shared_fitnesses(offspring,new_generation,sigma,alpha=1):
    #Sort individuals in decreasing order based on their shared fitness value
    for o in range(len(offspring)):
        dists = hamming_distance(offspring[o].order,new_generation)
        onePlusBeta = 1
        for d in dists:
            #Update onePlusBeta value if distance inferior to sigma threshold
            if d <= sigma:
                onePlusBeta += 1 - (d/sigma)**alpha
        offspring[o].shared_fitness = offspring[o].fitness * onePlusBeta  #Add  onePlusBeta penalty to fitness
        if o!= len(offspring)-1:
            if offspring[o].shared_fitness > offspring[o+1].shared_fitness: 
                #Early stopping : shared fitness can only decrease. if indiv i remains better than i+1 after update, there is no need in computing the next values
                break 
    return sorted(offspring, key=lambda sol: sol.shared_fitness, reverse=True)

def elimination(population, offspring, lamda):
    #lambda + mu elimination
    order = population + offspring
    order = sort_according_to_fitnesses(order)
    return order[0:lamda]

def elimination_fitness_sharing(population,offspring,lamda):
    #Apply fitness sharing based on already selected individuals
    new_pop = []
    #First iteration
    sigma_dict = {50:15,100:30,250:75,500:150,750:225,1000:300}
    sigma =sigma_dict[len(population[0].order)] 
    sorted = offspring+population
    sorted = [i for i in sorted if i.age <=5]  #Age based elimination
    sorted = sort_according_to_shared_fitnesses(sorted,new_pop,sigma)
    new_pop.append(sorted[0])
    sorted[0].age += 1
    sorted = sorted[1:] #Remove the selected offsprings
    #Mail loop
    while len(new_pop) < lamda:
        sorted = sort_according_to_shared_fitnesses(sorted,new_pop,sigma)
        new_pop.append(sorted[0])
        sorted[0].age +=1
        sorted = sorted[1:] #Remove the selected offspring
    
    return new_pop


def swap_mutate(individual,parameters):
    #Swap the positions of two cities, if it results in valid transitions
    tsp = individual.tsp
    if random.uniform(0, 1) <parameters.standard_alfa:
        swap_valid=False
        while not swap_valid:
            i = random.randint(1, len(individual.order) - 2)
            j = random.randint(1, len(individual.order) - 2)
            if individual.order[j] in tsp.valid_transition[individual.order[i-1]] and individual.order[i+1] in tsp.valid_transition[individual.order[j]]:
                if individual.order[i] in tsp.valid_transition[individual.order[j-1]] and individual.order[j+1] in tsp.valid_transition[individual.order[i]]:
                    individual.order[i], individual.order[j] = individual.order[j], individual.order[i]
                    swap_valid=True
    return individual


def insertion_mutate(individual,parameters):
    #Insert one city at a new index position, if it results in valid transitions
    tsp=individual.tsp
    if random.uniform(0,1) < parameters.standard_alfa:
        insertion_valid=False
        while not insertion_valid:
            i = random.randint(1, len(individual.order) - 2)
            j = random.randint(1, len(individual.order) - 2)
            smallest_index = min(i, j)
            biggest_index = max(i, j)
            city = individual.order[biggest_index]
            if individual.order[biggest_index+1] in tsp.valid_transition[individual.order[biggest_index-1]]:
                if individual.order[smallest_index+1] in tsp.valid_transition[city] and city in tsp.valid_transition[individual.order[smallest_index-1]]:
                    individual.order = np.delete(individual.order,biggest_index)
                    individual.order = np.concatenate((individual.order[:smallest_index], [city], individual.order[smallest_index:]))
                    insertion_valid=True
    return individual

def inversion_mutate(individual,parameters):
    #Invert a subpath
    if random.uniform(0, 1) < parameters.standard_alfa:
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        smallest_index = min(i, j)
        biggest_index = max(i, j)
        subset = individual.order[smallest_index:biggest_index]
        subset=np.flip(subset).astype('int')
        individual.order = np.append(individual.order[:smallest_index], np.append(subset,individual.order[biggest_index:]))        
    return individual


def OrderCrossover(mum, dad):
    '''
    '''
    dad_copy = dad.copy()
    nb_cities = len(mum)
    i = random.randint(0,nb_cities-1)
    j = random.randint(0,nb_cities-1)
    #initialise child_order with values that are not cities
    child_order = np.array(nb_cities * [-1])
    random_sublist = mum[min(i,j): max(i,j)]
    child_order[min(i,j):max(i,j)] = random_sublist
    dad_copy=np.array([elem for elem in dad_copy if elem not in random_sublist],dtype="int")

    for index in range(len(child_order)):
        if child_order[index] == -1:
            child_order[index], dad_copy = dad_copy[0], dad_copy[1:]
    return [child_order]

def CycleCrossover(mum, dad):
    '''
    '''
    nb_cities = len(mum)
    dad_copy = dad.copy()
    child_order = np.array(nb_cities * [-1])
    idx = random.randint(0,nb_cities-1)
    child_order[idx] = mum[idx]
    no_cycle=True
    while no_cycle:
        new_city = dad[idx]
        idx = np.where(mum ==dad[idx])[0]
        if new_city in child_order:
            no_cycle =False
        else:
            child_order[idx] = new_city

    dad_copy=np.array([elem for elem in dad_copy if elem not in child_order],dtype="int")
    for index in range(len(child_order)):
        if child_order[index] == -1:
            child_order[index], dad_copy = dad_copy[0], dad_copy[1:]
    
    return [child_order]


def CX2(mum,dad):
    '''
    '''
    nb_cities = len(mum)
    dad_copy = dad.copy()
    child_order = np.array(nb_cities * [-1])
    idx = random.randint(0,nb_cities-1)
    child_order[idx] = mum[idx]
    no_cycle=True
    while no_cycle:
        new_city = dad[np.where(mum==dad[idx])[0]]
        idx = np.where(mum==dad[np.where(mum ==dad[idx])[0]])[0]
        if new_city in child_order:
            no_cycle =False
        else:
            child_order[idx] = new_city
    dad_copy=np.array([elem for elem in dad_copy if elem not in child_order],dtype="int")
    for index in range(len(child_order)):
        if child_order[index] == -1:
            child_order[index], dad_copy = dad_copy[0], dad_copy[1:]
    
    return [child_order]

def CSOX(p1,p2):
    '''
    Variant of the ordered crossover which generates 6 offsprings for each pair of parents.
    Source : 
    '''
    offsprings = {}
    nb_cities = len(p2)
    r1 = random.randint(1,nb_cities-4)
    r2 = random.randint(r1+2,nb_cities-2)
    for i in range(3):
        p1_copy = p1.copy()
        p2_copy = p2.copy()
        if i ==0:
            pos1,pos2 = r1,r2
        elif i==1:
            pos1,pos2=0,r1
        else:
            pos1,pos2=r2+1,nb_cities-1
        
        offsprings[2*i+1],offsprings[2*i+2] = np.array(nb_cities * [-1]),np.array(nb_cities * [-1])
        offsprings[2*i+1][pos1:pos2+1] = p1[pos1:pos2+1]
        offsprings[2*i+2][pos1:pos2+1] = p2[pos1:pos2+1]
        #Update parents copy
        p1_copy = [town for town in np.concatenate((p1[pos2+1:],p1[:pos2+1]))if town not in offsprings[2*i+2]]
        p2_copy = [town for town in np.concatenate((p2[pos2+1:],p2[:pos2+1])) if town not in offsprings[2*i+1]]
        for index in range(pos2+1,len(offsprings[2*i+1])):
            offsprings[2*i+1][index], p2_copy = p2_copy[0], p2_copy[1:]
            offsprings[2*i+2][index], p1_copy = p1_copy[0], p1_copy[1:]
        for index in range(pos1):
            offsprings[2*i+1][index], p2_copy = p2_copy[0], p2_copy[1:]
            offsprings[2*i+2][index], p1_copy = p1_copy[0], p1_copy[1:]    
            
    return offsprings.values()