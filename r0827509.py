import Reporter
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from numba import jit
import warnings
warnings.filterwarnings('ignore')



class Parameters:
    def __init__(self, lamda=100, mu=100, k=10, its=100,lower_bound=1,upper_bound=1,
                 standard_alfa=0.3, random_share=0.8,elimination='lambda+musharing',n_LSO=50):
        self.lamda = lamda
        self.its = its
        self.mu = mu
        self.k = k
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.standard_alfa = standard_alfa
        self.random_share = random_share
        self.elimination = elimination
        self.n_LSO = n_LSO

class r0827509:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)


    def optimize(self, 
                 filename,
                 parameters = Parameters(),
                 heuristic=True):
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        finalFitness = -10000000
        tsp = TravellingSalesmanProblem(distanceMatrix,parameters)
        # g_best=[]
        # g_mean=[]
        # g_time=[]
        yourConvergenceTestsHere = True


        while( yourConvergenceTestsHere):
            ##################
            # Initialization #
            ##################
            
            start_time = time.time()
            if heuristic:
                population = heur_initialize(tsp,parameters)
            else:
                population = initialize(tsp, parameters)
            for idx in range(0,min(parameters.n_LSO,len(population))):
                LSO(population[idx])
            # Evaluate fitness and reporting
            fitnesses = [indiv.fitness for indiv in population]
            bestObjective = max(fitnesses)
            bestSolution = population[fitnesses.index(bestObjective)]
            meanObjective=  mean(fitnesses)
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
                if i%3==0:
                    print('swap round')
                    island1, island2 = self.swap_islands(island1,island2,parameters,5)
                island1 = self.create_new_generation(tsp,island1,parameters,[swap_mutate,insertion_mutate], OrderCrossover)
                island2 = self.create_new_generation(tsp,island2,parameters,[swap_mutate,insertion_mutate], CSOX)
                population = island1 + island2
                # population = self.create_new_generation(tsp,population,parameters,[mutation.swap_mutate,mutation.insertion_mutate],
                                                        # crossover.CSOX,local_search,True)
                #Adaptation of the algo
                # Evaluate fitness and reporting
                fitnesses = [indiv.fitness for indiv in population]
                if bestObjective == max(fitnesses):
                    same_best_counter +=1
                else:
                    same_best_counter=0
                bestObjective = max(fitnesses)
                bestSolution = population[fitnesses.index(bestObjective)]
                meanObjective = mean(fitnesses)


                # g_best.append([abs(int(bestObjective))])
                # g_mean.append([abs(int(meanObjective))])
                # g_time.append([time.time()-start_time])

                print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)


                timeLeft = self.reporter.report(meanObjective, bestObjective, np.array(bestSolution.order))
                if timeLeft < 0:
                    print('No time left')
                    print('Current iter %s'%i)
                    print('Results after timeout')
                    print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
                    finalFitness = bestObjective
                    break

                if same_best_counter>=50:
                    print('Same best for 50 iterations')
                    finalFitness = bestObjective
                    break

                if i == parameters.its:
                    print('Optimization finished')
                    print('Elapsed time : %s seconds'%(time.time()-start_time))
                    print('Results after all iterations')
                    print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
                    
                    finalFitness = bestObjective
                    break
            
            yourConvergenceTestsHere = False

        return finalFitness

    
    def create_new_generation(self,tsp,population,parameters,mutation,recombination,local_search=True,elitism = True):
        """
        Takes a population as input and update it using recombination, mutation, local search, and elimination
        """
        elimination_dict = {'lambda+mu':elimination,
                            'lambda+musharing':elimination_fitness_sharing,
                            }
        # mut_op = random.choice(mutation) #Randomly select one mutation among all the available ones
        elimination_op = elimination_dict[parameters.elimination]
        offspring = list()
        if elitism:
            elite = elimination(population,offspring,1,parameters.k) 
            population.remove(elite[0])
        else:
            elite = []
        for _ in range(1, parameters.mu): #Offsprings for this iteration
            p_1 = selection(population, parameters.k)
            p_2 = selection(population, parameters.k)
            for order in recombination(p_1.order,p_2.order,tsp):
                child = CandidateSolution(tsp,parameters, order=order)
                for mut_op in mutation:
                    child = mut_op(child,tsp)
                child.computeFitness()
                offspring.append(child)
       
        #Mutation of seed population,
        mutated_population = list()
        for ind in population:
            for mut_op in mutation:
                ind = mut_op(ind,tsp)
            ind.computeFitness()
            mutated_population.append(ind)
        population = mutated_population
         #To make sure that the best solution remains whatever
        population = [elite[0]] + elimination_op(population,offspring, len(population),5)
        
        #Local search best individuals
        if local_search:
            for idx in range(0,min(parameters.n_LSO,len(population))): #Make the number of LSO vary depending on the problem
                #The population is sorted, we make local search on the best one
                LSO(population[idx])
        return population

    def swap_islands(self,island1,island2,parameters,indiv_to_swap=5):
        #Move individuals between islands
        moveto1 = [selection(island2,parameters.k) for _ in range(indiv_to_swap)]
        moveto2 = [selection(island1,parameters.k) for _ in range(indiv_to_swap)]
        island1 = [i for i in island1 if i not in  moveto2]
        island1 += moveto1
        island2 = [i for i in island2 if i not in  moveto1]
        island2 += moveto2

        return island1, island2



class TravellingSalesmanProblem:
    def __init__(self, distance_matrix,parameters):
        self.number_of_cities = len(distance_matrix)
        #Represent distance matrix as a dictionary
        distance_dict = dict()
        j = 0
        self.valid_transition = {}  #Dictionary of all valid transitions (makes search  for initial solutions much faster)
        while j < len(distance_matrix):
            distance_dict[j] = {index: distance_matrix[j][index] for index in range(len(distance_matrix[j]))}
            self.valid_transition[j] = {index for index in range(len(distance_matrix[j])) if not math.isinf(distance_matrix[j][index]) and distance_matrix[j][index]}
            j += 1 #Test we ban all distances bigger than 5000 as invalid
        self.distance_matrix = distance_dict


class CandidateSolution:
    @jit
    def __init__(self, travelling_salesman_problem, parameters, order=None):
        self.age=0
        self.alfa = parameters.standard_alfa 
        self.tsp = travelling_salesman_problem
        self.shared_fitness = 0
        taboo = set()
        cities = set(range(0,self.tsp.number_of_cities))
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
            penalty +=1
        distance += last_dist
        
        if math.isinf(distance):
            #Convert infinite to a very large negative number 
            distance =  1e10
        self.fitness = -distance  - (penalty * 1e3)


""" Randomly initialize the population """
def initialize(tsp, lamda,standard_alfa=0.1):
    population = list()
    for _ in range(lamda):
        indiv = CandidateSolution(tsp, standard_alfa)
        indiv.computeFitness()
        population.append(indiv)
    return population

""" Heuristically initialize the population """
@jit
def heur_initialize(tsp, parameters):
    random_pop = list()
    greedy_pop = list()
    n_random = round(parameters.lamda * parameters.random_share)
    n_greedy = parameters.lamda - n_random
    count = 0
    cities = set(range(0,tsp.number_of_cities))
    taboo = set()
    while count < n_random:
        #Standard initialization
        indiv = CandidateSolution(tsp,parameters)
        count +=1
        indiv.computeFitness()
        random_pop.append(indiv)

    for _ in range(min(100,tsp.number_of_cities)):
        # Make up to 100 Greedy initialization, each time with a different starting position
        heur_parameter = random.uniform(parameters.lower_bound, parameters.upper_bound)
        current_city = random.choice(list(cities - taboo))
        # print(current_city)
        taboo.add(current_city)  #Add starting cities to the taboo for the greedy heuristic
        order = [current_city]
        timer = time.time()
        while len(order) < tsp.number_of_cities-1 and time.time() - timer < 10:
            #Force path to be legal
            if len(set(tsp.valid_transition[current_city])-set(order)) == 0:
                order = order[:-10] 
                current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order])
                order.append(current_city)
            elif random.uniform(0, 1) > heur_parameter:
                # choose next_city random --> never happens in practice
                current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order])
                order.append(current_city)
            else:
                current_city = min(set(tsp.valid_transition[order[-1]]) - set(order), key=tsp.distance_matrix[current_city].__getitem__)
                order.append(current_city)
        if not time.time() - timer > 10:
            remaining_city = [number for number in range(0,tsp.number_of_cities) if number not in order]
            order.append(remaining_city[0])
            order = np.array(order)
            indiv = CandidateSolution(tsp, parameters, order) #0.1
            indiv.computeFitness()
            greedy_pop.append(indiv)
    
    print(len(greedy_pop))
    fitnesses = [indiv.fitness for indiv in greedy_pop]
    # print(fitnesses)
    sorted_fitnesses = sorted(fitnesses,reverse=True)
    # print(sorted_fitnesses)
    greedy_pop =  [greedy_pop[fitnesses.index(f)]  for f in sorted_fitnesses[:n_greedy]]
    print(len(greedy_pop))
        
    return greedy_pop + random_pop


def mean(list):
    return sum(list)/len(list)

@jit
def LSO(indiv):
    # 2-opt local search
    start_time = time.time()
    #Case where i = 0
    subpath_cost = 0
    inverted_subpath_cost=0
    for j in range(1,len(indiv.order)-1):
        subpath=indiv.order[0:j+1]
        subpath_cost += indiv.tsp.distance_matrix[subpath[-2]][subpath[-1]]
        inverted_subpath = subpath[::-1]
        inverted_subpath_cost += indiv.tsp.distance_matrix[inverted_subpath[0]][inverted_subpath[1]]  #Add the new edge
        #Check first of all that the inverted subpath has no infinite cost, if it is the case we can ignore the rest
        if not math.isinf(inverted_subpath_cost):
            old_cost =  indiv.tsp.distance_matrix[indiv.order[-1]][indiv.order[0]] + indiv.tsp.distance_matrix[indiv.order[j]][indiv.order[j+1]]  + subpath_cost
            new_cost =  indiv.tsp.distance_matrix[indiv.order[0]][indiv.order[j+1]]  + indiv.tsp.distance_matrix[indiv.order[-1]][indiv.order[j]] + inverted_subpath_cost
            if new_cost < old_cost and not math.isinf(new_cost):
                indiv.order[0:j+1] = inverted_subpath
        else:
            #All larger inverted paths will also have infinite cost, we can break an pick another i
            break

    for i in range(1,len(indiv.order)-2):
        subpath_cost = 0
        inverted_subpath_cost=0
        if time.time() -start_time > 10:
          break
        for j in range(i+1,len(indiv.order)-1):
            subpath=indiv.order[i:j+1]
            subpath_cost += indiv.tsp.distance_matrix[subpath[-2]][subpath[-1]]
            inverted_subpath =subpath[::-1]
            inverted_subpath_cost += indiv.tsp.distance_matrix[inverted_subpath[0]][inverted_subpath[1]]
            #Check first of all that the inverted subpath has no infinite cost, if it is the case we can ignore the rest
            if not math.isinf(inverted_subpath_cost):
                old_cost =  indiv.tsp.distance_matrix[indiv.order[i-1]][indiv.order[i]] + indiv.tsp.distance_matrix[indiv.order[j]][indiv.order[j+1]]  + subpath_cost
                new_cost =  indiv.tsp.distance_matrix[indiv.order[i]][indiv.order[j+1]]  + indiv.tsp.distance_matrix[indiv.order[i-1]][indiv.order[j]] + inverted_subpath_cost
                if new_cost < old_cost and not math.isinf(new_cost):
                    indiv.order[i:j+1] = inverted_subpath
            else:
                #All larger inverted paths will also have infinite cost, we can break an pick another i
                break
    
    subpath_cost=0
    inverted_subpath_cost=0
    for j in range(0,len(indiv.order)-2):
        subpath = np.concatenate((indiv.order[-1:],indiv.order[0:j+1]))
        subpath_cost += indiv.tsp.distance_matrix[subpath[-2]][subpath[-1]]
        inverted_subpath =subpath[::-1]
        inverted_subpath_cost += indiv.tsp.distance_matrix[inverted_subpath[0]][inverted_subpath[1]]
        #Check first of all that the inverted subpath has no infinite cost, if it is the case we can ignore the rest
        if not math.isinf(inverted_subpath_cost):
            old_cost =  indiv.tsp.distance_matrix[indiv.order[-2]][indiv.order[-1]] + indiv.tsp.distance_matrix[indiv.order[j]][indiv.order[j+1]]  + subpath_cost
            new_cost =  indiv.tsp.distance_matrix[indiv.order[-1]][indiv.order[j+1]]  + indiv.tsp.distance_matrix[indiv.order[-2]][indiv.order[j]] + inverted_subpath_cost
            if new_cost < old_cost and not math.isinf(new_cost):
                indiv.order[-1] = inverted_subpath[0]
                indiv.order[0:j+1] = inverted_subpath[1:]
        else:
            #All larger inverted paths will also have infinite cost, we can break
            break

        
    indiv.computeFitness()



def selection(population, k):
    """ k-tournament selection """
    selected = random.sample(population, k)
    fitnesses = [indiv.fitness for indiv in selected] 
    return selected[fitnesses.index(max(fitnesses))]


def sort_according_to_fitnesses(candidate_solutions_list):
    return sorted(candidate_solutions_list, key=lambda sol: sol.fitness, reverse=True)


def hamming_distance(ind1,new_generation):
    #Compute the hamming distance between one offspring and the selected population
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
    for o in range(len(offspring)):
        dists = hamming_distance(offspring[o].order,new_generation)
        onePlusBeta = 1
        for d in dists:
            if d <= sigma:
                onePlusBeta += 1 - (d/sigma)**alpha
        offspring[o].shared_fitness = offspring[o].fitness * onePlusBeta  #Add penalty to fitness
        if o!= len(offspring)-1:
            if offspring[o].shared_fitness > offspring[o+1].shared_fitness:  #Whatever happens, this time, the best remains the best. so we can stop
                break  #This saves computation time (fitness value can only decrease, so we can stop early)
        #Based on the fact that the offsprings are sorted by shared fitness as input
    return sorted(offspring, key=lambda sol: sol.shared_fitness, reverse=True)

def elimination(population, offspring, lamda,k):
    #lambda + mu elimination
    order = population + offspring
    order = sort_according_to_fitnesses(order)
    return order[0:lamda]

def elimination_fitness_sharing(population,offspring,lamda,k):
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
    sorted = sorted[1:]
    while len(new_pop) < lamda:
        sorted = sort_according_to_shared_fitnesses(sorted,new_pop,sigma)
        new_pop.append(sorted[0])
        sorted[0].age +=1
        sorted = sorted[1:] #Remove the selected offspring
    
    return new_pop


def swap_mutate(individual,tsp):
    if random.uniform(0, 1) < individual.alfa:
        swap_valid=False
        while not swap_valid:
            i = random.randint(1, len(individual.order) - 2)
            j = random.randint(1, len(individual.order) - 2)
            if individual.order[j] in tsp.valid_transition[individual.order[i-1]] and individual.order[i+1] in tsp.valid_transition[individual.order[j]]:
                if individual.order[i] in tsp.valid_transition[individual.order[j-1]] and individual.order[j+1] in tsp.valid_transition[individual.order[i]]:
                    individual.order[i], individual.order[j] = individual.order[j], individual.order[i]
                    swap_valid=True
    return individual


def insertion_mutate(individual,tsp):

    if random.uniform(0,1) < individual.alfa:
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

def inversion_mutate(individual,tsp):
    if random.uniform(0, 1) < individual.alfa:

        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        smallest_index = min(i, j)
        biggest_index = max(i, j)
        subset = individual.order[smallest_index:biggest_index]
        subset=np.flip(subset).astype('int')
        individual.order = np.append(individual.order[:smallest_index], np.append(subset,individual.order[biggest_index:]))        
    return individual


def OrderCrossover(mum, dad,tsp):
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

def CycleCrossover(mum, dad,tsp):
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

@jit
def CX2(mum,dad,tsp):
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

def CSOX(p1,p2,tsp):
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