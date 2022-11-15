import random
import numpy as np
"""
Procedure: OX
1. Select a substring from a parent at random.
2. Produce a proto-child by copying the substring into the
corresponding position of it.
3. Delete the cities which are already in the substring from the 2nd
parent. The resulted sequence of cities contains the cities that the
proto-child needs.
4. Place the cities into the unfixed positions of the proto-child from left
to right according to the order of the sequence to produce an
offspring.
"""
#Jonathan : can't we create two offsprings each time? we select the same i and j but inverse the roles of mum and dad for the 2nd one
def OrderCrossover(mum, dad):
    # print('mum', mum)
    # print('dad', dad)
    # Do not change the real dad but take a copy instead
    dad_copy = dad.copy()
    nb_cities = len(mum)
    i = random.randint(0,nb_cities-1)
    j = random.randint(0,nb_cities-1)
    #initialise child_order with values that are not cities
    child_order = np.array(nb_cities * [-1])
    random_sublist = mum[min(i,j): max(i,j)]
    child_order[min(i,j):max(i,j)] = random_sublist
    dad_copy=np.array([elem for elem in dad_copy if elem not in random_sublist],dtype="int")

    for index in range(len(child_order)):
        if child_order[index] == -1:
            child_order[index], dad_copy = dad_copy[0], dad_copy[1:]
    return child_order


# mum = [1,2,3, 4, 5, 6, 7, 8, 9]
# dad = [5,7, 4, 9, 1, 3, 6, 2, 8]
# print(OrderCrossover(mum,dad))