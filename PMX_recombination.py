import random


def PMX_recombination(parent1,parent2):
        #Define crossover points
        random. seed(101)
        crossover_point_1=round(len(parent1)*random.random())
        #print(f"crossover_point_1 {crossover_point_1}")
        random.seed(123)
        crossover_point_2=round(len(parent1)*random.random())
        # print(f"crossover_point_2 {crossover_point_2}")
        offspring=[None]*len(parent1)
        first=min(crossover_point_1,crossover_point_2)
        i=first
        final=max(crossover_point_1,crossover_point_2)
        #Paste segment of Parent 1 between crossover points into offspring
        position_in_2=[]
        while i<=final:
            offspring[i]=parent1[i]
            #Find the index in P2 of the elements in the segment of P1 that are also in P2
            if offspring[i] in parent2[:first] or offspring[i] in parent2[final+1:]:
           #     if parent2.index(offspring[i])<first or parent2.index(offspring[i])>final:
               position_in_2.append(parent2.index(offspring[i]))         
            i+=1
        #print(f"offspring step 1 {offspring}")
        #Find the elements of the segment in P2 that are not in the segment of P1
        not_in_1_just_2=[]      
        for x in parent2[first:final+1]:
                if x not in offspring:
                    not_in_1_just_2.append(x)
        #Putting the elementS of P2 not in the segment of P1 into the position of the elements of P2 that are in the segment of P1
        l=0
        #print(f"position_in_2 {position_in_2}")
        #print(f"not_in_1_just_2 {not_in_1_just_2}")
        for k in position_in_2:
            if l<len(not_in_1_just_2):
                offspring[k]=not_in_1_just_2[l]
                l+=1
        #print(f"offspring step 2 {offspring}")
        #Copying all the elements of P2 that are still not in offspring into the exact position in offspring as they are in P2
        for m in parent2:
            if m not in offspring:
                #print(f"m {m}")
                #print(f"offspring(index(m)) {offspring[parent2.index(m)]}")
                offspring[parent2.index(m)]=m
                #print(f"offspring(index(m)) after {offspring[parent2.index(m)]}")
        #print(f"offspring step 3 {offspring}")
        return offspring

# p=PMX()
# parent1=[0,1,2,3,4,5,6,7,8,9]
# parent2=[3,4,7,8,5,6,9,0,1,2]
# descendant=p.PMX_recombination(parent1,parent2)
# print(descendant)





        





