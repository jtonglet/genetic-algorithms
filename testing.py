from r0827509 import r0827509, Parameters
p50 = Parameters() 
p100 = Parameters() 
p250 = Parameters() 
p500 = Parameters() 
p750 = Parameters() 
p1000 = Parameters() 
reporter = r0827509()
# reporter.optimize('tour50.csv',p50)  #Optimal : 54119, heuristic : 66540  , My score: 54440  OK
# reporter.optimize('tour100.csv',p100) #Optimal : 85807, heuristic :  103436, My score: 85808 OK Reached global minima!!!
# reporter.optimize('tour250.csv',p250) #Optimal : 331990, heuristic : 405662 , My score : 337829 OK
reporter.optimize('tour500.csv',p500) #Optimal : 60911, heuristic : 78579, My score :  70295 OK 
# reporter.optimize('tour750.csv',p750)  #Optimal : 101258, heuristic : 134752  , My score :  130320 OK
# reporter.optimize('tour1000.csv',p1000) #Optimal :50604, heuristic : 75446, My score : 71415 OK

