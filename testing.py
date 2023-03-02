#Evaluate results
import r0827509

reporter=r0827509.r0827509() 

reporter.optimize('data/tour50.csv')  #Optimal : 54119, heuristic : 66540  , My score: 54440 
reporter.optimize('data/tour100.csv') #Optimal : 85807, heuristic :  103436, My score: 85808 
# reporter.optimize('data/tour250.csv') #Optimal : 331990, heuristic : 405662 , My score : 337829 
# reporter.optimize('data/tour500.csv') #Optimal : 60911, heuristic : 78579, My score :  70295
# reporter.optimize('data/tour750.csv')  #Optimal : 101258, heuristic : 134752  , My score :  130320 
# reporter.optimize('data/tour1000.csv') #Optimal :50604, heuristic : 75446, My score : 71415 
