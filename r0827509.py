import Reporter
import numpy as np
import random
import math
import mutation
import selection
import order_crossover
import time
# import matplotlib.pyplot as plt
from numba import jit



class Parameters:
    def __init__(self, lamda, mu, k, its,lower_bound,upper_bound,standard_alfa,random_proba):
        self.lamda = lamda
        self.its = its
        self.mu = mu
        self.k = k
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.standard_alfa = standard_alfa
        self.random_proba = random_proba


class r0827509:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename,
                 parameters = Parameters(lamda=400, mu=100, k=2, its=500, lower_bound = 0.5,upper_bound=0.8,standard_alfa=0.1,
                                        random_proba=0.8),
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
                                             parameters.random_proba)
            else:
                population = initialize(tsp, parameters.lamda,parameters.standard_alfa)
        
            # Evaluate fitness and reporting
            fitnesses = [indiv.fitness for indiv in population]
            bestObjective = max(fitnesses)
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

                if i%50==0:
                    island1, island2 = self.swap_islands(island1,island2,parameters.lamda//20)

                #Idea : different crossover for each island
                if i%5==0:
                    local_search = True
                else:
                    local_search = False
                island1 = self.create_new_generation(tsp,island1,order_crossover.OrderCrossover,selection.lambdamu_elimination,parameters,local_search)
                island2 = self.create_new_generation(tsp,island2,order_crossover.CycleCrossover,selection.lambdamu_elimination,parameters,local_search)
                population = island1 + island2
                # Evaluate fitness and reporting
                #The best objective and best solutions are at the front of the population after the elimination
                fitnesses = [indiv.fitness for indiv in population]
                if bestObjective == max(fitnesses):
                    same_best_counter +=1
                else:
                    same_best_counter=0
                bestObjective = max(fitnesses)
                bestSolution = population[fitnesses.index(max(fitnesses))]
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

    
    def create_new_generation(self,tsp,population,crossover,elimination,parameters,local_search=True):
        """
        Take a population as input and update it using recombination, mutation, local search, and elimination
        """

        offspring = list()
        for _ in range(1, parameters.mu): #Offsprings for this iteration
            p_1 = selection.selection(population, parameters.k)
            p_2 = selection.selection(population, parameters.k)
            child = CandidateSolution(tsp,standard_alfa=parameters.standard_alfa, order=crossover(p_1.order, p_2.order))
            child.computeFitness()
            mut = mutation.inversion_mutate(child)
            if local_search:
                LSO(mut,tsp)
            offspring.append(mut)
        #Mutation of seed population
        # mutated_population = list()
        # for ind in population:
        #     mut = mutation.inversion_mutate(ind)
        #     mutated_population.append(mut)
        # population = mutated_population
        # Elimination
        # return selection.elimination(population, offspring, len(population))
        return elimination(population,offspring, len(population))

    def swap_islands(self,island1,island2,indiv_to_swap=5):
        #Move individuals between islands
        # moveto1 =  []
        # moveto2 = []
        # for _ in range(indiv_to_swap):
            # moveto1.append(selection.selection(island2, parameters.k))
            # moveto2.append(selection.selection(island1, parameters.k))
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
    def __init__(self, travelling_salesman_problem, standard_alfa=0.1, order=None):
        self.alfa = max(0.01, standard_alfa + 0.02 * np.random.normal())
        self.tsp = travelling_salesman_problem
        if order is None:
            #Do better than pure random, try to guarantee legal path
            current_city = random.randint(0,self.tsp.number_of_cities-1)
            order = [current_city]
            while len(order) < self.tsp.number_of_cities-1:
                if len(set(self.tsp.valid_transition[current_city])-set(order)) == 0:
                    last_ten = order[:-10]
                    order = order[:-10] #Backtrack from two cities
                    current_city = random.choice([city for city in self.tsp.valid_transition[order[-1]] if city not in order +last_ten])
                    order.append(current_city)
                else:
                # choose next_city random
                    current_city = random.choice([city for city in self.tsp.valid_transition[order[-1]] if city not in order])
                    order.append(current_city)
                
            remaining_city = [number for number in range(0,self.tsp.number_of_cities) if number not in order]
            order.append(remaining_city[0])

            self.order = np.array(order)
        else:
            self.order = order
        # for number in range(self.tsp.number_of_cities):
        #     assert number in self.order

    
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
def initialize(tsp, lamda,standard_alfa=0.1):
    population = list()
    for _ in range(lamda):
        indiv = CandidateSolution(tsp, standard_alfa)
        indiv.computeFitness()
        population.append(indiv)
    return population

""" Heuristically initialize the population """
#This part is very slow, takes about 2 minutes just for 250t
#Ideas : greedy heuristic but on a random sample only, better python structre for the distance matrix

@jit
def heur_initialize(tsp, lamda,standard_alfa=0.1,lower_bound=0.5,upper_bound=0.8,random_proba=0.8):
    print('heuristic initalization')
    population = list()
    distance_matrix_copy = tsp.distance_matrix
    for _ in range(lamda):
        if random.uniform(0,1) > random_proba:  #0.9 is the threshold to use the heuristic initialization, otherwise it is random initialization
            # print('okay')
            heur_parameter = random.uniform(lower_bound, upper_bound)  #Generate a different one for each indiv
            current_city = random.randint(0,tsp.number_of_cities-1)
            order = [current_city]

            while len(order) < tsp.number_of_cities-1:

                if len(set(tsp.valid_transition[current_city])-set(order)) == 0:
                    last_ten = order[:-10]
                    order = order[:-10] #Backtrack from two cities
                    current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order +last_ten])
                    order.append(current_city)


                elif random.uniform(0, 1) > heur_parameter:
                    # choose next_city random
                    current_city = random.choice([city for city in tsp.valid_transition[order[-1]] if city not in order])
                    order.append(current_city)
                else:
                    current_city = min(set(tsp.valid_transition[order[-1]]) - set(order), key=distance_matrix_copy[current_city].__getitem__)
                    order.append(current_city)

            remaining_city = [number for number in range(0,tsp.number_of_cities) if number not in order]
            order.append(remaining_city[0])

            order = np.array(order)
            indiv = CandidateSolution(tsp, standard_alfa, order) #0.1
        else:
            #Standard initialization
            indiv = CandidateSolution(tsp,standard_alfa)
        #Add the individual to the population
        indiv.computeFitness()
        population.append(indiv)
    return population


def mean(list):
    return sum(list)/len(list)

@jit
def LSO(indiv,tsp):
    # 2-opt local search
    bestIndividual = indiv

    for i in range(0,len(indiv.order)-2):
        #Insert loop into the first position
        # print(indiv_copy.order)
        indiv_copy = CandidateSolution(tsp,order=indiv.order)
        indiv_copy.order[i], indiv_copy.order[i+1] = indiv.order[i+1], indiv_copy.order[i]
        # indiv_copy.order[1:i+1] = indiv.order[0:i]
        # indiv_copy.order[i+1:] = indiv.order[i+1:]
        #Instead of creating copies, only create it if the changes in the 3 arcs results in a cost reduction !!
        #That's why we need to use the distance matrix, to not compute the full cost each time
        indiv_copy.computeFitness()
        if indiv_copy.fitness > bestIndividual.fitness:
            bestIndividual.order = indiv_copy.order
            bestIndividual.fitness = indiv_copy.fitness
            # print('LSO has improved results')

    indiv.order = bestIndividual.order
    indiv.computeFitness()




#HRM mut, l 200, k 2n its 2000 and greedy --> 300k on 500t
p50 = Parameters(lamda=200, mu=1000, k=2, its=5000,lower_bound = 0.8,upper_bound=1,standard_alfa=0.1,random_proba=0.8) # Perfect with lamda, mu
p100 = Parameters(lamda=200, mu=1000, k=2, its=5000,lower_bound=0.8,upper_bound=1,standard_alfa=0.1,random_proba=0.8) #Perfect with lamda, mu
p250 = Parameters(lamda=100, mu=500, k=2, its=5000,lower_bound=0.8,upper_bound=1,standard_alfa=0.3,random_proba=0.8) #Perfect with lamda,mu
p500 = Parameters(lamda=50, mu=125, k=2, its=5000,lower_bound=0.8,upper_bound=1,standard_alfa=0.1,random_proba=0.7) #Very very good (80k)
p1000 = Parameters(lamda=100, mu=250, k=2, its=5000,lower_bound=0.8,upper_bound=1,standard_alfa=0.1,random_proba=0.8) #Very very good (80k)
reporter = r0827509()

# reporter.optimize('tour50.csv',p50)
reporter.optimize('tour100.csv',p100)
# reporter.optimize('tour250.csv',p250)
# reporter.optimize('tour500.csv',p500)
# reporter.optimize('tour750.csv',p500)
# reporter.optimize('tour1000.csv',p500)

#Initialization is now very very good and almost on par with benchmark
#What to do :
#Mix of greedy and random initialization
# Improve diversity promotion
#Use a good local search
#Better recombination