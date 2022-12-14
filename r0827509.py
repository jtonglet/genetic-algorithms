import Reporter
import numpy as np
import random
import math
import mutation
import selection
import crossover
import time
# import matplotlib.pyplot as plt
from numba import jit
import warnings
warnings.filterwarnings('ignore')



class Parameters:
    def __init__(self, lamda=50, mu=50, k=2, its=50,lower_bound=0.5,upper_bound=0.8,standard_alfa=0.1,
                random_share=0.8,elimination='lambda+mu'):
        self.lamda = lamda
        self.its = its
        self.mu = mu
        self.k = k
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.standard_alfa = standard_alfa
        self.random_share = random_share
        self.elimination = elimination

class r0827509:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename,
                 parameters = Parameters(lamda=400, mu=100, k=2, its=500, lower_bound = 0.5,upper_bound=0.8,standard_alfa=0.1,
                                        random_share=0.8),
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
                population = heur_initialize(tsp,parameters.lamda,parameters.standard_alfa,
                                             parameters.lower_bound,parameters.upper_bound,
                                             parameters.random_share)
            else:
                population = initialize(tsp, parameters.lamda,parameters.standard_alfa)
            for idx in range(0,min(50,len(population))): #Make the number of LSO vary depending on the problem   #To be adjusted
                #The population is sorted, we make local search on the best one
                LSO(population[idx])
                # print('Done')
            # population = [LSO(ind) for ind in population]
            # Evaluate fitness and reporting
            fitnesses = [indiv.fitness for indiv in population]
            bestObjective = max(fitnesses)
            distance = 0
            bestSolution = population[fitnesses.index(bestObjective)]
            # for i in range(len(bestSolution.order)-1):
            #     distance = tsp.distance_matrix[bestSolution.order[i]][bestSolution.order[i+1]]
            #     print(distance)
            # print( tsp.distance_matrix[bestSolution.order[-1]][bestSolution.order[0]])
            # print(len(bestSolution.order))
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

                # Idea : different crossover for each island
                if i%1==0:
                    # print('LSO round')
                    local_search = True
                else:
                    local_search = False
                island1 = self.create_new_generation(tsp,island1,parameters,[mutation.swap_mutate,mutation.insertion_mutate],
                                                        crossover.OrderCrossover,local_search)
                island2 = self.create_new_generation(tsp,island2,parameters,[mutation.swap_mutate,mutation.insertion_mutate],
                                                        crossover.CSOX,local_search)
                population = island1 + island2
                # print(len(population))
                # population = self.create_new_generation(tsp,population,parameters,[mutation.swap_mutate,mutation.insertion_mutate],
                                                        # crossover.CSOX,local_search,True)
                #Adaptation of the algo
                parameters.standard_alfa -= parameters.standard_alfa * 1/parameters.its
                # print(parameters.standard_alfa)
                # Evaluate fitness and reporting
                #The best objective and best solutions are at the front of the population after the elimination
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

                if i == parameters.its: #The end is reached
                    print('Optimization finished')
                    print('Elapsed time : %s seconds'%(time.time()-start_time))
                    print('Results after all iterations')
                    print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
                    
                    finalFitness = bestObjective
                    break
            
            #Either the max num of iterations is reached or the time left is up
            yourConvergenceTestsHere = False

        return finalFitness

    
    def create_new_generation(self,tsp,population,parameters,mutation,recombination,local_search=True,elitism = True):
        """
        Take a population as input and update it using recombination, mutation, local search, and elimination
        """
        elimination_dict = {'lambda+mu':selection.elimination,
                            'lambdamu':selection.lambdamu_elimination,
                            'k-tournament':selection.k_tournament_elimination,
                            'lambda+musharing':selection.elimination_fitness_sharing,
                            'lambdamusharing':selection.lambdamu_elimination_fitness_sharing,
                            'k-tournamentsharing':selection.k_tournament_elimination_fitness_sharing
                            }
        # mut_op = random.choice(mutation) #Randomly select one mutation among all the available ones
        elimination_op = elimination_dict[parameters.elimination]
        offspring = list()
        if elitism:
            elite = selection.elimination(population,offspring,1,parameters.k) 
            population.remove(elite[0])
            # elite = [mut_op(elite[0])]
            # n = len(population)-1
        else:
            elite = []
            # n=len(population)
        for _ in range(1, parameters.mu): #Offsprings for this iteration
            p_1 = selection.selection(population, parameters.k)
            p_2 = selection.selection(population, parameters.k)
            for order in recombination(p_1.order,p_2.order,tsp):
                child = CandidateSolution(tsp,standard_alfa=parameters.standard_alfa, order=order)
                child.computeFitness()
                # print(child.fitness)
                # if child.fitness > p_1.fitness:
                    # print(child.fitness)
                for mut_op in mutation:
                    child = mut_op(child,tsp)
                child.computeFitness()
                # print(child.fitness)
                # print(' ')
                offspring.append(child)
        if parameters.elimination in ['lambda+mu','lambda+musharing']:
        #Mutation of seed population,
            mutated_population = list()
            for ind in population:
                for mut_op in mutation:
                    ind = mut_op(ind,tsp)
                ind.computeFitness()
                mutated_population.append(ind)
            population = mutated_population
         #To make sure that the best solution remains whatever
        population = elite + elimination_op(population,offspring, len(population),5)
        # print(len(population))
        
        #Local search best individuals
        if local_search:
            # start_time = time.time()
            for idx in range(0,min(50,len(population))): #Make the number of LSO vary depending on the problem
                #The population is sorted, we make local search on the best one
                LSO(population[idx])
            # print('Time needed for LSO : %s'%(time.time()-start_time))
        return population

    def swap_islands(self,island1,island2,parameters,indiv_to_swap=5):
        #Move individuals between islands
        moveto1 = [selection.selection(island2,parameters.k) for _ in range(indiv_to_swap)]
        moveto2 = [selection.selection(island1,parameters.k) for _ in range(indiv_to_swap)]
        island1 = [i for i in island1 if i not in  moveto2]
        island1 += moveto1
        island2 = [i for i in island2 if i not in  moveto1]
        island2 += moveto2

        return island1, island2

class TravellingSalesmanProblem:
    def __init__(self, distance_matrix,parameters):
        self.number_of_cities = len(distance_matrix)
        #represent distance matrix as a dictionary
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
    def __init__(self, travelling_salesman_problem, standard_alfa=0.1, order=None):
        self.age=0
        self.alfa = standard_alfa #max(0.1, standard_alfa + 0.005 * np.random.normal())
        self.tsp = travelling_salesman_problem
        self.shared_fitness = 0
        taboo = set()
        cities = set(range(0,self.tsp.number_of_cities))
        if order is None:
            #Do better than pure random, try to guarantee legal path
            # current_city = random.choice(list(cities - taboo))
            # taboo.add(current_city)
            # order = [current_city]
            # while len(order) < self.tsp.number_of_cities-1:
            #     # choose next_city random
            #     if len(set(self.tsp.valid_transition[current_city])-set(order)) == 0:
            #         current_city = random.choice([city for city in range(self.tsp.number_of_cities) if city not in order])
            #         order.append(current_city)

            #     else:
            #         current_city = random.choice([city for city in self.tsp.valid_transition[order[-1]] if city not in order])
            #         order.append(current_city)     
            # remaining_city = [number for number in range(0,self.tsp.number_of_cities) if number not in order]
            # order += remaining_city
            order = list(range(0,self.tsp.number_of_cities))
            random.shuffle(order)
            self.order = np.array(order)
            # print(len(self.order))
        else:
            self.order = order
    
    def computeFitness(self):
        #Fitness is a parameter of the class
        #Penalty for infinite  costs segments
        distance = 0
        penalty = 0
        for i in range(len(self.order)-1):
            # print((self.order[i],self.order[i+1]))
            new_dist = self.tsp.distance_matrix[self.order[i]][self.order[i+1]]
            if math.isinf(new_dist):
                penalty += 1
            distance += new_dist
            # print(new_dist)
        last_dist = self.tsp.distance_matrix[self.order[-1]][self.order[0]]
        # print((self.order[-1],self.order[0]))
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
def heur_initialize(tsp, lamda,standard_alfa=0.1,lower_bound=0.5,upper_bound=0.8,random_share=0.8):
    population = list()
    n_random = round(lamda * random_share)
    count = 0
    cities = set(range(0,tsp.number_of_cities))
    taboo = set()
    while len(population) < lamda:
        if count < n_random:  #Make it deterministic instead
            #Standard initialization
            indiv = CandidateSolution(tsp,standard_alfa)
            count +=1
        else:
            #Greedy initialization
            heur_parameter = random.uniform(lower_bound, upper_bound)  #Generate a different one for each indiv
            current_city = random.choice(list(cities - taboo))
            taboo.add(current_city)  #Add starting cities to the taboo for the greedy heuristic
            order = [current_city]
            timer = time.time()
            while len(order) < tsp.number_of_cities-1 and time.time() - timer < 10:
                #Force path to be legal
                if len(set(tsp.valid_transition[current_city])-set(order)) == 0:
                    order = order[:-10] #Backtrack from two cities
                    current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order])
                    order.append(current_city)
                elif random.uniform(0, 1) > heur_parameter:
                    # choose next_city random
                    current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order])
                    order.append(current_city)
                else:
                    current_city = min(set(tsp.valid_transition[order[-1]]) - set(order), key=tsp.distance_matrix[current_city].__getitem__)
                    order.append(current_city)
            if not time.time() - timer > 10:
                remaining_city = [number for number in range(0,tsp.number_of_cities) if number not in order]
                order.append(remaining_city[0])
                order = np.array(order)
                indiv = CandidateSolution(tsp, standard_alfa, order) #0.1
                count += 1
                #Add the individual to the population
        indiv.computeFitness()
        population.append(indiv)
        # print('indiv created')
    print('Intial population size %s '%len(population))
    return population


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
        #quadratic complexity
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
        # print(j)
        #Case where i is the last of the orders
        # print(indiv.order[-1:])
        # print(indiv.order[0:j+1])
        subpath = np.concatenate((indiv.order[-1:],indiv.order[0:j+1]))
        subpath_cost += indiv.tsp.distance_matrix[subpath[-2]][subpath[-1]]
        inverted_subpath =subpath[::-1]
        inverted_subpath_cost += indiv.tsp.distance_matrix[inverted_subpath[0]][inverted_subpath[1]]
        #Check first of all that the inverted subpath has no infinite cost, if it is the case we can ignore the rest
        if not math.isinf(inverted_subpath_cost):
            old_cost =  indiv.tsp.distance_matrix[indiv.order[-2]][indiv.order[-1]] + indiv.tsp.distance_matrix[indiv.order[j]][indiv.order[j+1]]  + subpath_cost
            new_cost =  indiv.tsp.distance_matrix[indiv.order[-1]][indiv.order[j+1]]  + indiv.tsp.distance_matrix[indiv.order[-2]][indiv.order[j]] + inverted_subpath_cost
            if new_cost < old_cost and not math.isinf(new_cost):
                # print('okay')
                indiv.order[-1] = inverted_subpath[0]
                indiv.order[0:j+1] = inverted_subpath[1:]
        else:
            #All larger inverted paths will also have infinite cost, we can break an pick another i
            break

        
    indiv.computeFitness()