from r0827509 import r0827509, Parameters
#HRM mut, l 200, k 2n its 2000 and greedy --> 300k on 500t
p50 = Parameters(lamda=200, mu=200,lower_bound = 0.7,upper_bound=1,random_share=0.9,standard_alfa=0.3,elimination='k-tournament') # Perfect with lamda, mu
p100 = Parameters(lamda=200, mu=200,lower_bound=0.8,upper_bound=1,random_share=0.9,elimination='k-tournament') #Perfect with lamda, mu
p250 = Parameters(lamda=100, mu=100,lower_bound=0.8,upper_bound=1,standard_alfa=0.4,elimination='k-tournament') #Perfect with lamda,mu
p500 = Parameters(lamda=100, mu=200,lower_bound=1,upper_bound=1,standard_alfa=0.1,random_share=0,elimination='k-tournament') #Very very good (80k)
p750 = Parameters(lamda=200, mu=200,its=50, lower_bound=1,upper_bound=1,standard_alfa=0.1,random_share=0,elimination='lambdamu') #Top  
p1000 = Parameters(lamda=200, mu=200, its=50, lower_bound=1,upper_bound=1,standard_alfa=0.1,random_share=0,elimination='lambdamu') #Very very good (80k)
reporter = r0827509()
# reporter.optimize('tour50.csv',p50)  #Also solved with pure random initialization
# reporter.optimize('tour100.csv',p100)
# reporter.optimize('tour250.csv',p250)
reporter.optimize('tour500.csv',p500)
# reporter.optimize('tour750.csv',p750)
# reporter.optimize('tour1000.csv',p1000)

#Initialization is now very very good and almost on par with benchmark
#What to do :
# Fitness sharing --> compare to the already selected individuals (modification of the current sort)

#Work only on the expensive subtour (e.g. at the end, the last 20 towns cost the most for the heuristic, optimize the changes there!)
