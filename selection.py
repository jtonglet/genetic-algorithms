import random


""" k-tournament selection """
def selection(population, k):
    selected = random.sample(population, k)
    fitnesses = [indiv.fitness for indiv in selected] 
    return selected[fitnesses.index(max(fitnesses))]


def sort_according_to_fitnesses(candidate_solutions_list):
    # return sorted(candidate_solutions_list, key=lambda sol: fitness(travelling_salesman_problem, sol), reverse=True)
    return sorted(candidate_solutions_list, key=lambda sol: sol.fitness, reverse=True)

def elimination(population, offspring, lamda,k):
    #lambda + mu elimination
    order = population + offspring
    order = sort_according_to_fitnesses(order)
    return order[0:lamda]

def k_tournament_elimination(population,offspring,lamda,k):
    order = []
    for _ in range(lamda):
        selected = random.sample(offspring,k)
        fitnesses = [indiv.fitness for indiv in selected] 
        best = selected[fitnesses.index(max(fitnesses))]
        order.append(best)
        offspring.remove(best)
    return order




def lambdamu_elimination(population,offspring,lamda,k):
    #Keep only the offsprings : require mu much bigger than lamda
    order = sort_according_to_fitnesses(offspring)
    return order[0:lamda]





