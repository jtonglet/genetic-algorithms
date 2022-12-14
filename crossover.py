import random
import numpy as np
from numba import jit
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
def OrderCrossover(mum, dad,tsp):
    # Do not change the real dad but take a copy instead
    # print(len(np.unique(dad)))
    # print(len(np.unique(mum)))
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
    return [child_order]

def CycleCrossover(mum, dad,tsp):
    #Default is that it may just produce the original one$
    nb_cities = len(mum)
    dad_copy = dad.copy()
    child_order = np.array(nb_cities * [-1])
    idx = random.randint(0,nb_cities-1)
    child_order[idx] = mum[idx]
    no_cycle=True
    while no_cycle:
        new_city = dad[idx]
        idx = np.where(mum ==dad[idx])[0]
        if new_city in child_order:
            no_cycle =False
        else:
            child_order[idx] = new_city

    dad_copy=np.array([elem for elem in dad_copy if elem not in child_order],dtype="int")
    for index in range(len(child_order)):
        if child_order[index] == -1:
            child_order[index], dad_copy = dad_copy[0], dad_copy[1:]
    
    return [child_order]

@jit
def CX2(mum,dad,tsp):
    nb_cities = len(mum)
    dad_copy = dad.copy()
    child_order = np.array(nb_cities * [-1])
    idx = random.randint(0,nb_cities-1)
    child_order[idx] = mum[idx]
    no_cycle=True
    while no_cycle:
        new_city = dad[np.where(mum==dad[idx])[0]]
        idx = np.where(mum==dad[np.where(mum ==dad[idx])[0]])[0]
        if new_city in child_order:
            no_cycle =False
        else:
            child_order[idx] = new_city

    dad_copy=np.array([elem for elem in dad_copy if elem not in child_order],dtype="int")
    for index in range(len(child_order)):
        if child_order[index] == -1:
            child_order[index], dad_copy = dad_copy[0], dad_copy[1:]
    
    return [child_order]

def CSOX(p1,p2,tsp):
    offsprings = {}
    nb_cities = len(p2)
    r1 = random.randint(1,nb_cities-4)
    r2 = random.randint(r1+2,nb_cities-2)
    # r1 = 2
    # r2=4
    for i in range(3):
        p1_copy = p1.copy()
        p2_copy = p2.copy()
        if i ==0:
            pos1,pos2 = r1,r2
        elif i==1:
            pos1,pos2=0,r1
        else:
            pos1,pos2=r2+1,nb_cities-1
        
        offsprings[2*i+1],offsprings[2*i+2] = np.array(nb_cities * [-1]),np.array(nb_cities * [-1])
        offsprings[2*i+1][pos1:pos2+1] = p1[pos1:pos2+1]
        offsprings[2*i+2][pos1:pos2+1] = p2[pos1:pos2+1]
        # print(offsprings[2*i+2][pos1:pos2+1])
        #Update parents copy
        p1_copy = [town for town in np.concatenate((p1[pos2+1:],p1[:pos2+1]))if town not in offsprings[2*i+2]]
        p2_copy = [town for town in np.concatenate((p2[pos2+1:],p2[:pos2+1])) if town not in offsprings[2*i+1]]
        for index in range(pos2+1,len(offsprings[2*i+1])):
            # if offsprings[2*i+1][pos2+1:][index] == -1:
            # print(index)
            offsprings[2*i+1][index], p2_copy = p2_copy[0], p2_copy[1:]
            offsprings[2*i+2][index], p1_copy = p1_copy[0], p1_copy[1:]
        for index in range(pos1):
            # if offsprings[2*i+1][pos2+1:][index] == -1:
            offsprings[2*i+1][index], p2_copy = p2_copy[0], p2_copy[1:]
            offsprings[2*i+2][index], p1_copy = p1_copy[0], p1_copy[1:]    
            
    return offsprings.values()


def SCX(p1,p2,tsp):
    #Requires the cost matrix, generates one offspring
    idx = random.choice(range(tsp.number_of_cities))
    town = p1[idx]
    order = [town]
    towns_set = set(range(tsp.number_of_cities))
    # print(towns_set)
    for _ in range(tsp.number_of_cities-1):
    
        town_pos1 = np.where(p1==town)[0][0]
        town_pos2 = np.where(p2==town)[0][0]
        if town_pos1==tsp.number_of_cities-1: #Last index
            candidate1=p1[0]
        else:
            candidate1 = p1[town_pos1+1]  #Next town in p1
            
        if town_pos2==tsp.number_of_cities-1: #Last index
            candidate2=p2[0]
        else:
            candidate2 = p2[town_pos2+1]  #Next town in p2

        if candidate1==candidate2 and not candidate1 in order:
            #Easiest situation : both lead to the same city
            town = candidate1
            order.append(town)
        else:
            if not candidate1 in order:
                cost1 = tsp.distance_matrix[town][candidate1]
            else:
                candidate1 = list(towns_set - set(order))[0]
                cost1 = tsp.distance_matrix[town][candidate1]
            if not candidate2 in order:
                cost2 = tsp.distance_matrix[town][candidate2]
            else:
                candidate2 = list(towns_set - set(order))[0]
                cost2 = tsp.distance_matrix[town][candidate2]
            
            if cost1 < cost2:
                town = candidate1
                order.append(town)
            else: 
                town = candidate2
                order.append(town)
    order = np.array(order)
    return [order]






# mum = np.array([3,5,8,2,1,9,4,6,7])
# dad = np.array([8,1,5,7,3,4,6,2,9])
# print(CSOX(mum,dad))
# print(CycleCrossover(mum,dad))
# print(CX2(mum,dad))