'''
A basic genetic algorithm for solving the binary KnapSack Problem
'''

import numpy as np
import random


class KnapSack:
    '''
    Representation of the problem instance
    Params:
        n_items (int): number of items to consider 
        value (array): array containing the value of each item 
        weight (array): array containing the weight of each item
        capacity (int): the maximum knapsack weight

    '''

    def __init__(self,
                 n_items=5,
                 capacity=20):
        self.n_items = n_items
        self.value = np.random.randint(low=1,high=10,size=n_items)
        self.weight = np.random.randint(low=1,high=capacity,size=n_items)
        self.capacity = capacity  


class Individual:
    '''
    Representation of an individual solution as a random permutation
    of the knapsack items.
    Params:
        problem (Knapsack): the problem instance 
        alpha (float): the mutation rate. Between 0 and 1.
        order (array): the order in which the items are filled in the knapsack. If None, initialized as a random permutation of the items.
    '''

    def __init__(self,
                 problem = KnapSack(),
                 alpha= 0.05,
                 order = None):

        self.problem = problem
        self.alpha = alpha  #Mutation rate
        self.order = order
        #No order predefined
        if not self.order:
            self.order = np.random.permutation(np.arange(self.problem.n_items))
    
    def compute_fitness(self):
        '''
        Compute the fitness (objective value) of the individual.
        '''
        fitness = 0
        cumulative_weight = 0
        for item in self.order: 
            #Reads the order from left to right, include items until capacity is reached
            cumulative_weight += self.problem.weight[item]
            if cumulative_weight <= self.problem.capacity: 
                fitness += self.problem.value[item] 
            else:
                break     
        return fitness

    def get_items_in_knapsack(self):
        '''
        Returns the set of items that are effectively included in the knapsack.
        '''
        cumulative_weight = 0
        item_set = set()
        for item in self.order:
            cumulative_weight += self.problem.weight[item]
            if cumulative_weight <= self.problem.capacity:
                 item_set.add(item)
            else:
                break
        return item_set

        
        
class evolutionaryAlgorithm:
    '''
    A generic evolutionary algorithm.
    Params:
        population_size (int): the number of individuals to generate in the initial population
        n_offsprings (int): number of offsprings to generate at each iteration of the algorithm
    '''

    def __init__(self,
                 population_size=200,
                 n_offsprings=100):

        self.population_size = population_size
        self.n_offsprings = n_offsprings
    
    def initialize_pop(self,
                       problem):
        '''
        Initialize randomly a population of individual for a problem.
        '''
        pop = [Individual(problem,random.uniform(0.001,0.05)) for _ in np.arange(self.population_size)]
        return pop

    def recombination(self,
                      p1,
                      p2,
                      t = 0.5,
                      crossover_function=None):
        '''
        Apply a crossover function to 2 parents to generate offsprings.
        '''

        items1 = p1.get_items_in_knapsack() 
        items2 = p2.get_items_in_knapsack()
        #All items in both knapsacks are directly included in the offspring
        offspring_items = list(items1.intersection(items2))

        #The items in the symmetric difference are randomly assigned if they pass a certain threshold t
        sym_dif = items1.symmetric_difference(items2)
        for item in sym_dif:
            if random.random() < t:
                offspring_items.append(item)

        #remaining items that did not appear or appeared in one parent and were not selected
        remaining_items = list(set([i for i in range(len(p1.order))]) - set(offspring_items))

        #Shuffle elements in the two lists
        random.shuffle(list(offspring_items)) 
        random.shuffle(list(remaining_items))
        #Combine the two lists
        offspring_order =  offspring_items + remaining_items
        #Create a new individual
        #The permutation probability alpha should be a recombiniation of those of the parents
        beta = 2 * random.random() -0.5
        offspring_alpha = p1.alpha + beta * (p2.alpha-p1.alpha)
        offspring = Individual(p1.problem,offspring_alpha,offspring_order) 
          
        return offspring
    
    def mutation(self,
                 indiv,
                 mutation_function=None):
        '''
        Apply a mutation operation to an individual.
        To save memory, keep the same individual object but change its 
        order parameter. Does not return anything
        '''

        #Swapping the places of 2 items in the knapsack.
        if random.random() < indiv.alpha:
            i,j = np.random.randint(0,len(indiv.order),size=2)
            indiv.order[i],indiv.order[j] = indiv.order[j],indiv.order[i]

    def selection(self,
                  population,
                  K = 10):
        '''
        Select one individual within the population with K-tournament.
        '''

        sample = np.random.choice(population,size=K)
        idx = np.array([x.compute_fitness() for x in sample]).argmax()
        #Keep the individual with the best fitness in the sample 
        return sample[idx]      
    
    def elimination(self,
                    population,
                    new_pop_size):
        '''
        Reduce the size of the population after creating offsprings.
        Params:
            population (list): a list of individuals containing the initial population and the offsprings
            new_pop_size (int): the size of the new  population after elimination.
        '''
        if new_pop_size > len(population):
            raise ValueError('new population should be smaller or equal to the input population')
        fitness_values = [x.compute_fitness() for x in population]
        #Rank individuals according to their fitness and keep the best
        new_pop_idx = sorted(range(len(population)),key=lambda k : fitness_values[k],reverse=True)[0:new_pop_size]
        new_pop = [population[i] for i in new_pop_idx]

        return new_pop


    def optimize(self,
            problem,
            iterations=15):
        '''
        Evolutionary algorithm main loop. 
        Params:
            problem (KnapSack): the problem to optimize
            iterations (int): the number of iterations performed on the main loop
        '''
        #Initialize the population
        population = self.initialize_pop(problem)
        #We have to keep track of the best solution uptill now 
        pop_fitness = [i.compute_fitness() for i in population]
        best_individual = population[pop_fitness.index(max(pop_fitness))]
            
        #Main Loop
        for _ in range(iterations):
            offsprings = []
            for _ in range(self.n_offsprings): 
                #Recombination
                p1 = self.selection(population)
                p2 = self.selection(population)
                offsprings.append(self.recombination(p1,p2))
                #Mutation of the offsprings
                for o in offsprings:
                    self.mutation(o)
            
            #Mutations of the original pop
            for ind in population:
                self.mutation(ind) 
            
            #Elimination
            population = self.elimination(population+offsprings,len(population))
            #Evaluation and update of the best solution
            pop_fitness = [ind.compute_fitness() for ind in population]   
            iter_best_individual = population[pop_fitness.index(max(pop_fitness))]
            if iter_best_individual.compute_fitness() >= best_individual.compute_fitness():
                best_individual = iter_best_individual
            print('Current best fitness : %s'%best_individual.compute_fitness())
    
        return best_individual



def heuristic(problem=KnapSack()):
    '''
    A simple heuristic for the knapsack problem, which sorts items according to their value-weight ratio.
    Serves as a benchmark for the evolutionary algorithm performance.
    '''
    order = np.arange(0,problem.n_items)
    heuristic_order = sorted(range(len(order)),key=lambda k : problem.value[k]/problem.weight[k],reverse=True)
    heur_indiv = Individual(problem,0,heuristic_order)
    return heur_indiv

if __name__=='__main__':
    kp = KnapSack(n_items=100,capacity=50)
    ev = evolutionaryAlgorithm(population_size=400,n_offsprings=200)
    print('Problem Parameters')
    print('---- Value -----')
    print(kp.value)
    print('---- Weight -----')
    print(kp.weight)
    print('---- Capacity -----')
    print(kp.capacity)
    print('    ')
    print('    ')
    solution = ev.optimize(kp)
    print(solution.get_items_in_knapsack())
    print('EA solution : %s'%solution.compute_fitness())
    print('    ')
    print(heuristic(kp).get_items_in_knapsack())
    print('Heuristic solution : %s'%heuristic(kp).compute_fitness())