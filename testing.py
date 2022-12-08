from r0827509 import r0827509, Parameters
#HRM mut, l 200, k 2n its 2000 and greedy --> 300k on 500t
p50 = Parameters(lamda=200, mu=1000,lower_bound = 0.7,upper_bound=1,random_share=0.9,elimination='lambdamu') # Perfect with lamda, mu
p100 = Parameters(lamda=200, mu=1000,lower_bound=0.8,upper_bound=1,elimination='lambdamu') #Perfect with lamda, mu
p250 = Parameters(lamda=100, mu=500,lower_bound=0.8,upper_bound=1,standard_alfa=0.3,elimination='lambdamu') #Perfect with lamda,mu
p500 = Parameters(lamda=100, mu=500,lower_bound=0.8,upper_bound=1,standard_alfa=0.2,elimination='lambdamu') #Very very good (80k)
p750 = Parameters(lamda=50, mu=250, lower_bound=1,upper_bound=1,standard_alfa=0.2,mutation='HPRM',elimination='lambdamu') #Very very good 
p1000 = Parameters(lamda=20, mu=100,lower_bound=0.8,upper_bound=1,mutation='HPRM',elimination='lambdamu') #Very very good (80k)
reporter = r0827509()

# reporter.optimize('tour50.csv',p50)
# reporter.optimize('tour100.csv',p100)
# reporter.optimize('tour250.csv',p250)
# reporter.optimize('tour500.csv',p500)
# reporter.optimize('tour750.csv',p750)
# reporter.optimize('tour1000.csv',p1000)

# file = open('tour50.csv')
# distanceMatrix = np.loadtxt(file, delimiter=",")
# tsp = TravellingSalesmanProblem(distanceMatrix)
# ind1 = CandidateSolution(tsp,order=np.array([27, 40, 47,  1, 28 ,25,  9, 46, 42,  8, 49, 41, 48, 24, 44, 22, 39, 23, 12,  5, 37,  4,  2, 35, 17, 30,  6, 38, 10, 33,  7, 18, 29, 34,  3, 13, 26, 36, 11, 21, 15, 20, 31, 32, 14,  0, 45, 16,
#  43, 19,]
# ))
# ind1=CandidateSolution(tsp)
# ind1.computeFitness()
# print(ind1.order)
# print(ind1.fitness)
# ind1 = LSO(ind1,tsp)
# print(ind1.order)
# print(ind1.fitness)


#Initialization is now very very good and almost on par with benchmark
#What to do :
# Improve diversity promotion --> fitness sharing complexity problem :/ crowding?  k_tournament elimination?  / reintroduce the island models
#LSO search time problem is solved
#Better recombination --> SCX
