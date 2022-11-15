import random


# def fitness(travelling_salesman_problem, individual):
#     distance = 0
#     index = 0
#     while index < len(individual.order) - 1:
#         current_city = individual.order[index]
#         next_city = individual.order[index + 1]
#         distance += travelling_salesman_problem.distance_matrix[current_city][next_city]
#         index += 1

#     try:
#         distance += travelling_salesman_problem.distance_matrix[individual.order[-1]][individual.order[0]]
#     except Exception as e:
#         print('ECCPETIN5', e)
#         print(individual.order)
#     # print(distance)
#     return -distance

""" k-tournament selection """
def selection(population, k):
    selected = random.sample(population, k)
    fitnesses = [indiv.fitness for indiv in selected] 
    return selected[fitnesses.index(max(fitnesses))]


def sort_according_to_fitnesses(candidate_solutions_list):
    # return sorted(candidate_solutions_list, key=lambda sol: fitness(travelling_salesman_problem, sol), reverse=True)
    return sorted(candidate_solutions_list, key=lambda sol: sol.fitness, reverse=True)

def elimination(population, offspring, lamda):
    #lambda + mu elimination
    for element in offspring:
        population.append(element)
    order = sort_according_to_fitnesses(population)
    return order[0:lamda]




