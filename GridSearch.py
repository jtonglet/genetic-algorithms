'''
Grid search for hyperparameter tuning
'''
 
from r0123456 import r0123456, Parameters
import time

def GridSearch(tour, params_dict):
    with open("GridSearch "+tour +str(time.time()),"w+") as f:
        best_params = []
        max_fitness = -1000000000
        for l in params_dict['lamda']:
            for m in params_dict['mu']:
                for k in params_dict['k']:
                    for u in params_dict['upper_bound']:
                        reporter = r0123456()
                        print('Parameters : %s lamda, %s mu, %s k , %s upper b\n'%(l,m,k,u))
                        f.write('Parameters : %s lamda, %s mu, %s k , %s upper b\n'%(l,m,k,u))
                        params = Parameters(l,m,k,500,u)
                        start_time = time.time()
                        fitness = reporter.optimize(tour,params)
                        if time.time() - start_time > 300:
                            f.write('Timed out\n')
                        f.write('Fitness : %s  \n'%fitness)
                        if fitness > max_fitness:
                            max_fitness = fitness
                            best_params = [l,m,k,u]
                        f.write('\n')
        f.write('Best Fitness %s with best params %s  \n'%(max_fitness,best_params))
        f.close()




if __name__ == '__main__':
    params_dict = {}
    params_dict['lamda'] = [50,100,200,300,400,500]  
    params_dict['mu'] = [50,100,200,300,400,500]
    params_dict['k'] = [2,3,4,5,7,10,15]
    params_dict['upper_bound'] = [0.2,0.4,0.6,0.8,1]

    GridSearch('tour50.csv',params_dict)