from r0827509 import r0827509, Parameters, TravellingSalesmanProblem
# import numpy  as np
#HRM mut, l 200, k 2n its 2000 and greedy --> 300k on 500t
p50 = Parameters(lamda=100, mu=100,its=100,lower_bound = 1,upper_bound=1,standard_alfa=0.3,elimination='lambda+musharing') # Perfect
p100 = Parameters(lamda=100, mu=100,its=50,lower_bound=1,upper_bound=1,standard_alfa=0.3,elimination='lambda+musharing') #Perfect 
p250 = Parameters(lamda=100, mu=100,lower_bound=1,upper_bound=1,standard_alfa=0.3,elimination='lambda+musharing') #Perfect 
p500 = Parameters(lamda=100, mu=100,its=100,lower_bound=1,upper_bound=1,standard_alfa=0.3,random_share=0.8,elimination='lambda+musharing') #Perfect
p750 = Parameters(lamda=100, mu=100,k=10,its=100, lower_bound=1,upper_bound=1,standard_alfa=0.3,random_share=0.8,elimination='lambda+musharing') #Top  
p1000 = Parameters(lamda=100, mu=100, its=50, lower_bound=1,upper_bound=1,standard_alfa=0.3,random_share=0.8,elimination='lambda+musharing') #Very very good (80k)
reporter = r0827509()
# reporter.optimize('tour50.csv',p50)  #Optimal : 54119, heuristic : 66540  , My score: 54500  OK
# reporter.optimize('tour100.csv',p100) #Optimal : 85807, heuristic :  103436, My score: 87661 OK
# reporter.optimize('tour250.csv',p250) #Optimal : 331990, heuristic : 405662 , My score : 342960 OK
# reporter.optimize('tour500.csv',p500) #Optimal : 60911, heuristic : 78579, My score :  73550 OK 
reporter.optimize('tour750.csv',p750)  #Optimal : 101258, heuristic : 134752  , My score :  130404 OK (tune params now, depends on random_share a lot)
# reporter.optimize('tour1000.csv',p1000) #Optimal :50604, heuristic : 75446, My score : 73177 OK

#Initialization is now very very good and almost on par with benchmark
#What to do :
#Only small parameter tuning now : value of k , value of alpha, value of random_share are the most important ones!!!