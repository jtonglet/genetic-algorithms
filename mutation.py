import random
import numpy as np

def swap_mutate(individual):
    if random.uniform(0, 1) < individual.alfa:
        # print('swap mutation')
        # Randomly swap elements
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        city = individual.order[i]

        individual.order[i] = individual.order[j]
        individual.order[j] = city
    
    individual.computeFitness()
    return individual


def insert_mutate(individual):

    if random.uniform(0,1) < individual.alfa:
        #print('insert mutation')

        # print('order before insert mutation', individual.order)
        # Randomly swap elements
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        smallest_index = min(i, j)
        city = individual.order.pop(max(i, j))
        # print('selected', tsp, 'to move next to', individual.order[smallest_index])
        individual.order = individual.order[:smallest_index] + [city] + individual.order[smallest_index:]

        individual.computeFitness()
    return individual


def scramble_mutate(individual):
    if random.uniform(0, 1) < individual.alfa:
        #print('scramble mutation')

        # print('order before scramble mutation', individual.order)
        # Randomly choose begin and index of subset to scramble
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        smallest_index = min(i, j)
        biggest_index = max(i, j)
        subset = individual.order[smallest_index:biggest_index]
        random.shuffle(subset)
        subset = np.array(subset)
        individual.order = np.append(individual.order[:smallest_index], np.append(subset,individual.order[biggest_index:]))  
        individual.computeFitness() 
    return individual


def inversion_mutate(individual):
    if random.uniform(0, 1) < individual.alfa:

        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        smallest_index = min(i, j)
        biggest_index = max(i, j)
        subset = individual.order[smallest_index:biggest_index]
        subset=np.flip(subset).astype('int')
        individual.order = np.append(individual.order[:smallest_index], np.append(subset,individual.order[biggest_index:]))        
        individual.computeFitness() #Update the fitness of the individual
    return individual

"""
Choose two mutation points a and b such that 1 ≤ a ≤ b ≤ n;
Repeat
Permute (xa, xb);
a = a + 1;
b = b − 1;
until a<b
"""
def permute(individual, a, b):
    subset = individual.order[a:b]
    subset=np.flip(subset).astype('int')
    individual.order = np.append(individual.order[:a], np.append(subset,individual.order[b:])) 

    return individual


def RSM(individual):
    if random.uniform(0, 1) < individual.alfa:
        i = random.randint(0, len(individual.order) - 1)
        j = random.randint(0, len(individual.order) - 1)
        a = min(i,j)
        b = max(i,j)
        while not a > b:
            individual = permute(individual, a, b)
            a += 1
            b -= 1
        individual.computeFitness()  # Update the fitness of the individual
    return individual


"""
Input: Parents x=[x1,x2,……,xn]and Pm is Mutation
probability
Output: Children x=[x1,x2,……,xn]
-----------------------------------------------------------
Choose two mutation points a and b such that 1 <= a <= b <= n
n;
Repeat
 Permute (xa, xb);
 Choose p a random number between
 if p < Pm then
 Choose j a random number between
 Permute (xa, xj);
End if
 a = a + 1;
 b = b − 1;
Until a < b 
"""
def HPRM(individual):
    if random.uniform(0, 1) < individual.alfa:
        i = random.randint(1, len(individual.order) - 1)
        j = random.randint(1, len(individual.order) - 1)
        a = min(i,j)
        b = max(i,j)
        while not a > b:

            individual = permute(individual,a,b)
            if random.uniform(0, 1) < individual.alfa:
                j = random.randint(1,len(individual.order)-1)
                individual = permute(individual,min(a,j), max(i,j))
            a += 1
            b -= 1
        individual.computeFitness()  # Update the fitness of the individual
    return individual



