#Implementation of Edge CrossOver

import numpy as np
import random


def EdgeCrossover(p1, p2):

    #Edge table Construction
    edge_table = {}

    for town in p1[:-1]:

        if p2.index(town) != len(p2) -1:
            edge_table[town] = [p1[p1.index(town)-1],p1[p1.index(town)+1],
                            p2[p2.index(town)-1],p2[p2.index(town)+1]]
        else:
            edge_table[town] = [p1[p1.index(town)-1],p1[p1.index(town)+1],
                            p2[-2],p2[0]]
    if p2[-1]!=p1[-1]:
        edge_table[p1[-1]] = [p1[-2],p1[0],p2[p2.index(p1[-1])-1],p2[p2.index(p1[-1])+1]]
    else: #Both are at the end
        edge_table[p1[-1]] = [p1[-2],p1[0],p2[-2],p2[0]]

    #The table is correct



    offspring = []
    #Choose a random starting point
    current_town = random.choice(p1)
    offspring.append(current_town)

    #Main Loop
    while len(offspring) < len(p1):

        for k,v in edge_table.items():
            #Update the table
            edge_table[k] = [town for town in v if town != current_town]

        if len(edge_table[current_town]) == 0:
            #print('Town %s has no path : %s'%(current_town,edge_table[current_town]))
            #Nothing more to open, go to tail
            current_town = random.choice(list(set(p1) - set(offspring)))
            offspring.append(current_town)

        elif len(edge_table[current_town]) == 1:
            current_town=edge_table[current_town][0]
            #print('Adding only option : %s'%current_town)

            offspring.append(current_town)

        elif len(edge_table[current_town]) != len(set(edge_table[current_town])):
            #There are dual paths
            for town in edge_table[current_town]:
                if edge_table[current_town].count(town) ==2:
                    #print('Dual link found for town %s : %s'%(current_town,town))
                    #Dual Link
                    current_town = town

                    offspring.append(current_town)
            
        else:
        #Shortest list (len of the set because there could be doubles)
            list_length = [len(set(edge_table[town])) for  town in edge_table[current_town]]
            min_length = min(list_length)
            if list_length.count(min_length) > 1:
                #ties : random choice
                idx = list_length.index(random.choice([i for i in list_length if i==min_length]))
                current_town = edge_table[current_town][idx]
                offspring.append(current_town)
            else:
                idx = list_length.index(min_length)
                current_town = edge_table[current_town][idx]
                offspring.append(current_town)


    return offspring



