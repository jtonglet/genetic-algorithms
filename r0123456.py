import Reporter
import numpy as np
import random
import math
import mutation
import selection
import PMX_recombination
import order_crossover
import CX2_crossover
import time
import matplotlib.pyplot as plt



class Parameters:
    # lamda = population size
    # mu = offspring size
    # k = for k-tournament selection
    # its = number of iterations

    def __init__(self, lamda, mu, k, its,upper_bound):
        self.lamda = lamda
        self.its = its
        self.mu = mu
        self.k = k
        self.upper_bound = upper_bound



# Modify the class name to match your student number.
class r0123456:

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
        #print(distanceMatrix)
        #print('size:', len(distanceMatrix), '-', len(distanceMatrix[0]))
        #print('distance 0-1', distanceMatrix[0][1])
        tsp = TravellingSalesmanProblem(distanceMatrix)
        g_best=[]
        g_mean=[]
        g_time=[]
        yourConvergenceTestsHere = True
        while( yourConvergenceTestsHere):

            #Initialize population
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
                    #print(p_1.order)
                    #print(p_2.order)
                    #child = CandidateSolution(tsp,alfa=0.1, order=PMX_recombination.PMX_recombination(p_1.order, p_2.order))
                    child = CandidateSolution(tsp,standard_alfa=0.1, order=order_crossover.OrderCrossover(p_1.order, p_2.order))
                    child.computeFitness()
                    mutated = mutation.inversion_mutate(child)
                    # print(mutated.fitness)

                    offspring.append(mutated)
                    #print(mutated.order)
                    #print('')
                    if np.array_equal(p_1.order,p_2.order) and np.array_equal(p_1.order,child.order):
                        no_diff += 1
                # print('no difference between parents and child', no_diff)


                #Mutation of seed population
                mutated_population = list()
                for ind in population:
                    mutated_population.append(mutation.inversion_mutate(ind))
                population = mutated_population


                # Elimination
                population = selection.elimination(population, offspring, parameters.lamda)

                # Evaluate fitness and reporting
                #The best objective and best solutions are at the front of the population after the elimination
                bestObjective = max([indiv.fitness for indiv in population])
                bestSolution = population[0]
                meanObjective = mean([indiv.fitness for indiv in population])

                g_best.append([abs(int(bestObjective))])
                g_mean.append([abs(int(meanObjective))])
                g_time.append([time.time()-start_time])

                if no_diff > 197:
                    #Stop algorithm and report current best objective
                    print('algorithm stopped because each individual in the population was the same')
                    return bestObjective

                # print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective)

                # Your code here.

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

        # fig = plt.figure(1, dpi=300)
        # plt.title('Convergence graph')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Fitness')
        # plt.plot(g_time[10:], g_mean[10:], label="Mean fitness")
        # plt.plot(g_time, g_best, label="Best fitness")
        # plt.legend(loc="upper right")   
        # plt.tight_layout()
        # plt.show()

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
    def __init__(self, travelling_salesman_problem, standard_alfa, order=None):
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
            # print(distance)
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
        self.fitness = -distance  - (penalty * 5e3)




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

def heur_initialize(tsp, lamda,upper_bound=0.8): #0.8
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

        indiv = CandidateSolution(tsp, 0.1, order)
        indiv.computeFitness()

        population.append(indiv)
    return population


def mean(list):
    # TODO - should return mean, not average
    return sum(list)/len(list)



# file = open('tour50.csv')
# distanceMatrix = np.loadtxt(file, delimiter=",")
# # file.close()
# tsp = TravellingSalesmanProblem(distanceMatrix)
# indiv = CandidateSolution(tsp)
# indiv.computeFitness()
# print(indiv.fitness)


#HRM mut, l 200, k 2n its 2000 and greedy --> 300k on 500t
# parameters = Parameters(lamda=2000, mu=400, k=10, its=500,upper_bound=0.8 )
# reporter = r0123456()
# reporter.optimize('tour250.csv',parameters)
# # a = CandidateSolution(tsp)