import random
import numpy as np
from scipy.spatial.distance import hamming
from numba import jit
import time


""" k-tournament selection """
def selection(population, k):
    selected = random.sample(population, k)
    fitnesses = [indiv.fitness for indiv in selected] 
    return selected[fitnesses.index(max(fitnesses))]


def sort_according_to_fitnesses(candidate_solutions_list):
    return sorted(candidate_solutions_list, key=lambda sol: sol.fitness, reverse=True)

# @jit
def hamming_distance(ind1,new_generation):
    #Compute the hamming distance between one offspring and the selected population
    #Code is correct
    dists = []
    for ind2 in new_generation:
        town_start = ind1[0]
        idx_start_2 = np.where(ind2.order==town_start)[0][0]
        sorted_ind2 = np.concatenate((ind2.order[idx_start_2:],ind2.order[:idx_start_2]))
        # print(ind1)
        # print(sorted_ind2)
        dist = 0
        for i in range(len(ind1)):
            if ind1[i] != sorted_ind2[i]:
                dist+=1
        # print(dist)
        dists.append(dist)
    return dists
    
    
def sort_according_to_shared_fitnesses(offspring,new_generation,sigma,alpha=1):
    for o in range(len(offspring)):
        # start= time.time()
        dists = hamming_distance(offspring[o].order,new_generation)
        onePlusBeta = 1
        for d in dists:
            #Compute one plus beta
            if d <= sigma:
                onePlusBeta += 1 - (d/sigma)**alpha
        # print(onePlusBeta)
        offspring[o].shared_fitness = offspring[o].fitness * onePlusBeta  #Add penalty to fitness
        # else: 
        #     offspring[o].shared_fitness = offspring[o].fitness
        # print('time taken to compute shared fitness %s'%(time.time()-start))
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
    # print(population[0].age)
    sorted = offspring+population
    sorted = [i for i in sorted if i.age <=5]  #Age based elimination
    sorted = sort_according_to_shared_fitnesses(sorted,new_pop,sigma)
    new_pop.append(sorted[0])
    sorted[0].age += 1
    sorted = sorted[1:]
    while len(new_pop) < lamda:
        # start = time.time()
        sorted = sort_according_to_shared_fitnesses(sorted,new_pop,sigma)
        #Every time the offsprings are sorted from best to worse according to shared fitness
        new_pop.append(sorted[0])
        sorted[0].age +=1
        sorted = sorted[1:] #Remove the selected offspring
        # print('Time taken for new indiv %s'%(time.time()-start))
    
    return new_pop



def k_tournament_elimination(population,offspring,lamda,k):
    order = []
    for _ in range(lamda):
        selected = random.sample(offspring,k)
        fitnesses = [indiv.fitness for indiv in selected] 
        best = selected[fitnesses.index(max(fitnesses))]
        order.append(best)
        offspring.remove(best)
    return order

def k_tournament_elimination_fitness_sharing(population,offspring,lamda,k):
#Apply fitness sharing based on already selected individuals
    new_pop = []
    #First iteration
    sigma_dict = {50:10,100:20,250:50,500:100,750:150,1000:200}
    sigma =sigma_dict[len(population[0].order)] 
    offspring = sort_according_to_shared_fitnesses(offspring,new_pop,sigma)
    new_pop.append(offspring[0])
    offspring[0].age +=1
    offspring = offspring[1:]
    while len(new_pop) < lamda:
        # start = time.time()
        offspring = sort_according_to_shared_fitnesses(offspring,new_pop,sigma)
        #Every time the offsprings are sorted from best to worse according to shared fitness
        new_pop.append(offspring[0])
        offspring[0].age += 1
        offspring = offspring[1:] #Remove the selected offspring
        # print('Time taken for new indiv %s'%(time.time()-start))

    return new_pop



def lambdamu_elimination(population,offspring,lamda,k):
    #Keep only the offsprings : require mu much bigger than lamda
    order = sort_according_to_fitnesses(offspring)
    return order[0:lamda]


def lambdamu_elimination_fitness_sharing(population,offspring,lamda,k):
    #Apply fitness sharing based on already selected individuals
    new_pop = []
    #First iteration
    sigma_dict = {50:20,100:40,250:100,500:200,750:300,1000:400}
    sigma =sigma_dict[len(population[0].order)] 
    offspring = sort_according_to_shared_fitnesses(offspring,new_pop,sigma)
    new_pop.append(offspring[0])
    offspring = offspring[1:]
    while len(new_pop) < lamda:
        # start = time.time()
        offspring = sort_according_to_shared_fitnesses(offspring,new_pop,sigma)
        #Every time the offsprings are sorted from best to worse according to shared fitness
        new_pop.append(offspring[0])
        offspring = offspring[1:] #Remove the selected offspring
        # print('Time taken for new indiv %s'%(time.time()-start))
    
    return new_pop







