'''
Implementation of the CX2 crossover for the path representation of the Traveling Salesman Problem.
Based on : "Genetic Algorithm for Traveling Salesman Problem with Modified Cycle Crossover Operator"
'''

def CX2Crossover(p1,p2):
    #The algorithm is implemented in a recursive way
    #Base case : the parent has only one town left

        
    #First town
    # print(p2[0])
    child1_order = [p2[0]]
    # print(p2[p1.index(p2[p1.index(p2[0])])])
    child2_order = [p2[p1.index(p2[p1.index(p2[0])])]]

    #Following towns
    for i in range(0,len(p1)-1):
        new_town1 = p2[p1.index(child2_order[i])]
        # print(new_town1)
        count = 1
        while new_town1 in child1_order:
            new_town1 = p2[p1.index(child2_order[i])+count]
            count += 1

        child1_order.append(new_town1)
        # print(p2[p1.index(p2[p1.index(child1_order[i+1])])])
        child2_order.append(p2[p1.index(p2[p1.index(child1_order[i+1])])])
        # else:
        #     #We have reached the exception point
        #     #Repeat the crossover on the smaller population
        #     break
    


    # reduced_p1 = [t for t in p1 if t  not in set(child2_order).intersection(set(p1))]
    # reduced_p2 = [t for t in p2 if t not in set(child2_order).intersection(set(p1))]
    # # print(len(reduced_p2)==len(reduced_p1))

    
    # # print(reduced_p1)
    # # print(reduced_p2)
    # # if len(reduced_p1)!=0:
    # child1_order += CX2Crossover(reduced_p1,reduced_p2)[0]
    # child2_order += CX2Crossover(reduced_p1,reduced_p2)[1]


    return child1_order,child2_order

#Case 1
# mum = [3,4,8,2,7,1,6,5]
# dad = [4,2,5,1,6,8,3,7]
#Case 2
mum = [1,2,3,4,5,6,7,8]
dad = [2,7,5,8,4,1,6,3]
#Case 3
mum = [47,  3, 24, 20, 21,  2, 11, 33, 44, 40, 46, 48,  5, 42, 35, 28, 6, 16, 45,  0,  4, 43,  1, 36, 15, 26, 25, 39, 37, 19, 27, 8, 38, 29, 12, 14, 31, 22,  7, 30, 49, 13, 32, 10, 34,  9, 18, 23, 17, 41]
dad = [39, 23, 18, 12, 47, 20, 21, 30, 19, 27, 37, 44, 38, 29,  8, 22, 0,  4, 48, 46, 32, 13, 28, 42, 34, 24,  3, 14, 43,  1,  9, 2, 11, 33,  7, 15, 26, 25, 40, 49,  5, 10, 35, 17, 36, 41,  6, 31, 45, 16]
offsprings = CX2Crossover(mum,dad)
# print(len(offsprings[0]))
# print(len(offsprings[1]))
# for i in range(50):
#     if i not in offsprings[0]:
#         print(i)

# print(len(offsprings[1]))
# print(len(offsprings[0]))
# print('Offspring 1 : %s  , offspring2 : %s'%(offsprings[0],offsprings[1]))

# mum = [47,  3, 24, 20, 21,  2, 11, 33, 44, 40, 46, 48,  5, 42, 35, 28, 6, 16, 45,  0,  4, 43,  1, 36, 15, 26, 25, 39, 37, 19, 27, 8, 38, 29, 12, 14, 31, 22,  7, 30, 49, 13, 32, 10, 34,  9, 18, 23, 17, 41]
# dad = [39, 23, 18, 12, 47, 20, 21, 30, 19, 27, 37, 44, 38, 29,  8, 22, 0,  4, 48, 46, 32, 13, 28, 42, 34, 24,  3, 14, 43,  1,  9, 2, 11, 33,  7, 15, 26, 25, 40, 49,  5, 10, 35, 17, 36, 41,  6, 31, 45, 16]
