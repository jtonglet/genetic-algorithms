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
    def __init__(self, lamda=100, mu=100, k=2, its=50,lower_bound=0.5,upper_bound=0.8,standard_alfa=0.1,
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
        tsp = TravellingSalesmanProblem(distanceMatrix)
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
        
            # Evaluate fitness and reporting
            fitnesses = [indiv.fitness for indiv in population]
            bestObjective = max(fitnesses)
            distance = 0
            bestSolution = population[fitnesses.index(bestObjective)]
            for i in range(len(bestSolution.order)-1):
                distance += tsp.distance_matrix[bestSolution.order[i]][bestSolution.order[i+1]]
                print(distance)
            print(distance + tsp.distance_matrix[bestSolution.order[-1]][bestSolution.order[0]])
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

                # if i%20==0:
                #     print('swap round')
                #     island1, island2 = self.swap_islands(island1,island2,parameters.lamda//20)

                #Idea : different crossover for each island
                if i%5==0:
                    # print('LSO round')
                    local_search = True
                else:
                    local_search = False
                # island1 = self.create_new_generation(tsp,island1,parameters,mutation.inversion_mutate,crossover.OrderCrossover,local_search)
                # island2 = self.create_new_generation(tsp,island2,parameters,mutation.scramble_mutate,crossover.CycleCrossover,local_search)
                # population = island1 + island2
                population = self.create_new_generation(tsp,population,parameters,mutation.inversion_mutate,crossover.OrderCrossover,local_search,True)
                parameters.standard_alfa -= parameters.standard_alfa * 1/parameters.its
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

                if same_best_counter>=20:
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
                            'k-tournament':selection.k_tournament_elimination
                            }

        elimination_op = elimination_dict[parameters.elimination]
        offspring = list()
        if elitism:
            elite = selection.elimination(population,offspring,1,parameters.k) 
            n = len(population)-1
        else:
            elite = []
            n=len(population)
        for _ in range(1, parameters.mu): #Offsprings for this iteration
            p_1 = selection.selection(population, parameters.k)
            p_2 = selection.selection(population, parameters.k)
            for order in crossover.CSOX(p_1.order,p_2.order).values():
                # child = CandidateSolution(tsp,standard_alfa=parameters.standard_alfa, order=recombination(p_1.order, p_2.order))
                child = CandidateSolution(tsp,standard_alfa=parameters.standard_alfa, order=order)
                child.computeFitness()
                mut = mutation(child)
                offspring.append(mut)
        if parameters.elimination == 'lambda+mu':
        #Mutation of seed population,
            mutated_population = list()
            for ind in population:
                mut = mutation(ind)
                mutated_population.append(mut)
            population = mutated_population
         #To make sure that the best solution remains whatever
        population = elite + elimination_op(population,offspring, n,5)
        
        #Local search best individuals
        if local_search:
            # start_time = time.time()
            for idx in range(0,10): #Make the number of LSO vary depending on the problem
                #The population is sorted, we make local search on the best one
                LSO(population[idx],tsp)
            # print('Time needed for LSO : %s'%(time.time()-start_time))
        return population

    def swap_islands(self,island1,island2,indiv_to_swap=5):
        #Move individuals between islands
        moveto1 = random.sample(island2,indiv_to_swap)
        moveto2 = random.sample(island1,indiv_to_swap)
        island1 = [i for i in island1 if i not in  moveto2]
        island1 += moveto1
        island2 = [i for i in island2 if i not in  moveto1]
        island2 += moveto2

        return island1, island2

class TravellingSalesmanProblem:
    def __init__(self, distance_matrix):
        self.number_of_cities = len(distance_matrix)
        #represent distance matrix as a dictionary
        distance_dict = dict()
        j = 0
        self.valid_transition = {}  #Dictionary of all valid transitions (makes search  for initial solutions much faster)
        while j < len(distance_matrix):
            distance_dict[j] = {index: distance_matrix[j][index] for index in range(len(distance_matrix[j]))}
            self.valid_transition[j] = {index for index in range(len(distance_matrix[j])) if not math.isinf(distance_matrix[j][index])}
            j += 1
        self.distance_matrix = distance_dict


class CandidateSolution:
    @jit
    def __init__(self, travelling_salesman_problem, standard_alfa=0.1, order=None):
        self.alfa = max(0.1, standard_alfa + 0.02 * np.random.normal())
        self.tsp = travelling_salesman_problem
        if order is None:
            #Do better than pure random, try to guarantee legal path
            current_city = random.randint(0,self.tsp.number_of_cities-1)
            order = [current_city]
            while len(order) < self.tsp.number_of_cities-1:
                # if len(set(self.tsp.valid_transition[current_city])-set(order)) == 0:
                #     last_ten = order[:-10]
                #     order = order[:-10] #Backtrack from two cities
                #     current_city = random.choice([city for city in self.tsp.valid_transition[order[-1]] if city not in order +last_ten])
                #     order.append(current_city)
                # else:
                # choose next_city random
                if len(set(self.tsp.valid_transition[current_city])-set(order)) == 0:
                    current_city = random.choice([city for city in range(self.tsp.number_of_cities-1) if city not in order])
                    order.append(current_city)

                else:
                    current_city = random.choice([city for city in self.tsp.valid_transition[order[-1]] if city not in order])
                    order.append(current_city)     
            remaining_city = [number for number in range(0,self.tsp.number_of_cities) if number not in order]
            order += remaining_city

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
    while len(population) < lamda:
        if count < n_random:  #Make it deterministic instead
            #Standard initialization
            indiv = CandidateSolution(tsp,standard_alfa)
            # count +=1
        else:
            #Greedy initialization
            heur_parameter = random.uniform(lower_bound, upper_bound)  #Generate a different one for each indiv
            current_city = random.randint(0,tsp.number_of_cities-1)
            order = [current_city]
            timer = time.time()
            while len(order) < tsp.number_of_cities-1 and time.time() - timer < 10:
                #Force path to be legal
                if len(set(tsp.valid_transition[current_city])-set(order)) == 0:
                    last_ten = order[:-10]
                    order = order[:-10] #Backtrack from two cities
                    current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order +last_ten])
                    order.append(current_city)
                if random.uniform(0, 1) > heur_parameter:
                    # choose next_city random
                    current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order])
                    order.append(current_city)
                else:
                    current_city = min(set(tsp.valid_transition[order[-1]]) - set(order), key=tsp.distance_matrix[current_city].__getitem__)
                    order.append(current_city)
            if not time.time() - timer > 20:
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


def subpath_cost(order,tsp):
    #Compute the fitness of a subsample[i,j] of the indiv
    #Is needed because the tsp is not symmetric, so reverting the central part also has a cost.
    distance = 0
    # print(order)
    for t in range(0,len(order)-1):
        # print(tsp.distance_matrix[order[t]][order[t+1]])
        cost = tsp.distance_matrix[order[t]][order[t+1]]
        if math.isinf(cost):
            #Quit early if infinite found
            distance =  math.inf
            break
        else:
            distance += cost
    return distance


@jit
def LSO(indiv,tsp):
    # 2-opt local search
    start_time = time.time()
    for i in range(1,len(indiv.order)-2):
        if time.time() -start_time > 10:
          break
        #quadratic complexity
        for j in range(i+1,len(indiv.order)-1):
            inverted_subpath = indiv.order[i:j+1][::-1]
            inverted_subpath_cost = subpath_cost(inverted_subpath,tsp)
            #Check first of all that the inverted subpath has no infinite cost, if it is the case we can ignore the rest
            if not math.isinf(inverted_subpath_cost):
                old_cost =  tsp.distance_matrix[indiv.order[i-1]][indiv.order[i]] + tsp.distance_matrix[indiv.order[j]][indiv.order[j+1]]  + subpath_cost(indiv.order[i:j+1],tsp)
                new_cost =  tsp.distance_matrix[indiv.order[i]][indiv.order[j+1]]  + tsp.distance_matrix[indiv.order[i-1]][indiv.order[j]] + inverted_subpath_cost
                if new_cost < old_cost and not math.isinf(new_cost):
                    indiv.order[i:j+1] = inverted_subpath
            else:
                #All larger inverted paths will also have infinite cost, we can break an pick another i
                break
              
    indiv.computeFitness()