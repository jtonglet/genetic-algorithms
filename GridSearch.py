'''
Grid search for hyperparameter tuning
'''
 
from r0827509 import r0827509, Parameters
import time

def GridSearch(tour, params_dict):
    with open("GridSearch "+tour +str(time.time())+'.txt',"w+") as f:
        best_params = []
        max_fitness = -1000000000
        for l in params_dict['lamda']:
            for a in params_dict['alpha']:
                for k in params_dict['k']:
                    if l==20 and k>5:
                        break
                    # for u in params_dict['']:
                    reporter = r0827509()
                    print('Parameters : %s lamda, %s alfa, %s k \n'%(l,a,k))
                    f.write('Parameters : %s lamda, %s alfa, %s k \n'%(l,a,k))
                    params = Parameters(l,5*l,k,100,a)
                    start_time = time.time()
                    fitness, _ = reporter.optimize(tour,params)
                    if time.time() - start_time > 300:
                        f.write('Timed out\n')
                    f.write('Fitness : %s  \n'%fitness)
                    if fitness > max_fitness:
                        max_fitness = fitness
                        best_params = [l,a,k]
                    f.write('\n')
        print('Best Fitness %s with best params %s  \n'%(max_fitness,best_params))
        f.write('Best Fitness %s with best params %s  \n'%(max_fitness,best_params))
        f.close()




if __name__ == '__main__':
    params_dict = {}
    params_dict['lamda'] = [20,50,100,150]  
    params_dict['alpha'] = [0.1,0.3,0.5]
    params_dict['k'] = [2,5,10,15]
    GridSearch('tour750.csv',params_dict)