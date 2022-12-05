import Reporter
import numpy as np
import random
import math
import mutation
import selection
import order_crossover
import time
import matplotlib.pyplot as plt
from numba import jit



class Parameters:
    def __init__(self, lamda, mu, k, its,upper_bound):
        self.lamda = lamda
        self.its = its
        self.mu = mu
        self.k = k
        self.upper_bound = upper_bound


class r0827509:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename,
                 parameters = Parameters(lamda=400, mu=100, k=2, its=500, upper_bound=0.8),
                 heuristic=True):
        # Read distance matrix from file.
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
                population = heur_initialize(tsp,parameters.lamda,parameters.upper_bound)
            else:
                population = initialize(tsp, parameters.lamda)
        
            # Evaluate fitness and reporting
            fitnesses = [indiv.fitness for indiv in population]
            bestObjective = max(fitnesses)
            meanObjective=  mean(fitnesses)
            print(0, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
            print('Time needed to initialize the population : %s seconds'%(time.time()-start_time))
            for i in range(1, parameters.its+1): #Main loop
                # Selection, recombination, and mutation
                offspring = list()
                no_diff = 0
                for _ in range(1, parameters.mu): #Offsprings for this iteration
                    p_1 = selection.selection(population, parameters.k)
                    p_2 = selection.selection(population, parameters.k)
                    child = CandidateSolution(tsp,standard_alfa=0.1, order=order_crossover.OrderCrossover(p_1.order, p_2.order))
                    child.computeFitness()
                    mutated = mutation.inversion_mutate(child)
                    # LSO(mutated)
                    offspring.append(mutated)


                #Mutation of seed population
                mutated_population = list()
                for ind in population:
                    mut = mutation.inversion_mutate(ind)
                    # LSO(mut,tsp)
                    mutated_population.append(mut)
                population = mutated_population


                # Elimination
                population = selection.elimination(population, offspring, parameters.lamda)

                # Evaluate fitness and reporting
                #The best objective and best solutions are at the front of the population after the elimination
                bestObjective = max([indiv.fitness for indiv in population])
                bestSolution = population[0]
                meanObjective = mean([indiv.fitness for indiv in population])

                # g_best.append([abs(int(bestObjective))])
                # g_mean.append([abs(int(meanObjective))])
                # g_time.append([time.time()-start_time])

                print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)


                # Call the reporter with:
                #  - the mean objective function value of the population
                #  - the best objective function value of the population
                #  - a 1D numpy array in the cycle notation containing the best solution
                #    with city numbering starting from 0
                timeLeft = self.reporter.report(meanObjective, bestObjective, np.array(bestSolution.order))
                if timeLeft < 0:
                    print('No time left')
                    print('Current iter %s'%i)
                    print('Results after timeout')
                    print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
                    finalFitness = bestObjective
                    break

                if i == parameters.its: #The end is reached
                    print('Optimization finished')
                    print('Elapsed time : %s seconds'%(time.time()-start_time))
                    print('Results after all iterations')
                    print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)
                    
                    finalFitness = bestObjective
            
            #Either the max num of iterations is reached or the time left is up
            yourConvergenceTestsHere = False

        # Your code here.
        return finalFitness



class TravellingSalesmanProblem:
    def __init__(self, distance_matrix):
        self.number_of_cities = len(distance_matrix)
        #represent distance matrix as a dictionary
        distance_dict = dict()
        j = 0
        while j < len(distance_matrix):
            distance_dict[j] = {index: distance_matrix[j][index] for index in range(len(distance_matrix[j]))}
            j += 1
        #self.distance_matrix = distance_dict
        self.distance_matrix = distance_matrix


class CandidateSolution:
    def __init__(self, travelling_salesman_problem, standard_alfa=0.1, order=None):
        self.alfa = max(0.01, standard_alfa + 0.02 * np.random.normal())
        self.tsp = travelling_salesman_problem
        if order is None:
            initial_state = np.arange(self.tsp.number_of_cities)
            np.random.shuffle(initial_state)
            self.order = initial_state
        else:
            self.order = order
        for number in range(self.tsp.number_of_cities):
            assert number in self.order

    
    def computeFitness(self):
        #Fitness is a parameter of the class
        #Penalty for infinite  costs segments
        distance = 0
        penalty = 0
        for i in range(len(self.order)-1):
            new_dist = self.tsp.distance_matrix[self.order[i]][self.order[i+1]]
            if math.isinf(new_dist):
                penalty += 1
            distance += new_dist
        try:
            last_dist = self.tsp.distance_matrix[self.order[-1]][self.order[0]]
            if math.isinf(last_dist):
                penalty +=1
            distance += last_dist
        except Exception as e:
            print(self.order)
        
        if math.isinf(distance):
            #Convert infinite to a very large negative number 
            distance =  1e10
        self.fitness = -distance  - (penalty * 1e3)


""" Randomly initialize the population """
def initialize(tsp, lamda):
    population = list()
    for _ in range(lamda):
        indiv = CandidateSolution(tsp, 0.1)
        indiv.computeFitness()
        population.append(indiv)
    return population

""" Heuristically initialize the population """
#This part is very slow, takes about 2 minutes just for 250t
#Ideas : greedy heuristic but on a random sample only, better python structre for the distance matrix

@jit
def heur_initialize(tsp, lamda,upper_bound=0.8):
    print('heuristic initalization')
    population = list()
    distance_matrix_copy = tsp.distance_matrix
    for _ in range(lamda):
        heur_parameter = random.uniform(0, upper_bound)  #Generate a different one for each indiv
        current_city = random.randint(0,tsp.number_of_cities-1)
        order = [current_city]

        while len(order) < tsp.number_of_cities-1:
            if random.uniform(0, 1) > heur_parameter:
                # choose next_city random
                current_city = random.choice([city for city in range(tsp.number_of_cities) if city not in order])
                order.append(current_city)
            else:
                current_city = min(set(range(0,tsp.number_of_cities)) - set(order), key=distance_matrix_copy[current_city].__getitem__)
                order.append(current_city)

        remaining_city = [number for number in range(0,tsp.number_of_cities) if number not in order]
        order.append(remaining_city[0])

        order = np.array(order)

        indiv = CandidateSolution(tsp, 0.1, order) #0.1
        indiv.computeFitness()
        population.append(indiv)
    return population


def mean(list):
    # TODO - should return mean, not average
    return sum(list)/len(list)

@jit
def LSO(indiv,problem):
    # 2-opt local search
    bestIndividual = indiv

    for i in range(0,len(indiv.order)-2):
        #Insert loop into the first position
        # print(indiv_copy.order)
        indiv_copy = CandidateSolution(problem,order=indiv.order)
        indiv_copy.order[i], indiv_copy.order[i+1] = indiv.order[i+1], indiv_copy.order[i]
        # indiv_copy.order[1:i+1] = indiv.order[0:i]
        # indiv_copy.order[i+1:] = indiv.order[i+1:]
        #Instead of creating copies, only create it if the changes in the 3 arcs results in a cost reduction !!
        #That's why we need to use the distance matrix, to not compute the full cost each time
        indiv_copy.computeFitness()
        if indiv_copy.fitness > bestIndividual.fitness:
            bestIndividual = indiv_copy

    indiv.order = bestIndividual.order
    indiv.computeFitness()




#HRM mut, l 200, k 2n its 2000 and greedy --> 300k on 500t
parameters = Parameters(lamda=100, mu=100, k=2, its=5000,upper_bound=0.8 )
reporter = r0827509()
reporter.optimize('tour50.csv',parameters)