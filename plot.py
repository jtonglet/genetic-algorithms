
#Compute variation in values with a boxplot on tour 50 (100 iterations)
from r0827509 import r0827509, Parameters
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def print_minimum(tour):
    df = pd.read_csv(tour).replace(0,1000000000)
    dist = 0
    df
    for val in df.min(axis=1):
        dist += val
    
    print(dist)

def boxplot(tours,repetitions=100):
    #Compare variability in results on different tours
    #Idea : plot the heuristic value as a line as well
    fitnesses = []
    type  = []
    params = Parameters()
    for t in tours:
        for i in range(repetitions):
            print('Iteration : %s'%i)
            reporter = r0827509()
            best, mean = reporter.optimize(t,params)
            fitnesses += [best,mean]
            type += ['best','mean']

        data = pd.DataFrame({'fitnesses':fitnesses,
                            'type':type})
        data.to_csv('6-variability of results for '+t,index=False)

    sns.histplot(x=data['fitnesses'],hue=data['type'])
    plt.title('Variability of the best and mean fitness for tour 50')
    plt.xticks(rotation = 45)
    plt.plot()
    plt.savefig('boxplot_tour_50.png',bbox='tight',dpi=300)

def histplot(tours_df):
    data = pd.read_csv(tours_df[0])
    for dataset in tours_df[1:]:
        df = pd.read_csv(dataset)
        data = data.append(df, ignore_index=True)

    copy = data.copy(deep=True)
    data = data.append(copy,ignore_index=True)
    data['fitnesses']= data['fitnesses'].abs()
    means = data[data.type=='mean']
    best = data[data.type=='best']
    print('best')
    print('average')
    print(best.mean(axis=0))
    print('standard deviation')
    print(best.std())
    print('means')
    print('average')
    print(means.mean(axis=0))
    print('standard deviation')
    print(means.std())

    sns.histplot(x=data['fitnesses'],hue=data['type'])
    plt.title('Variability of the best and mean fitness for tour 50')
    plt.xticks(rotation = 45)
    plt.plot()
    plt.savefig('histplot_tour_50.png',bbox='tight',dpi=300)

    


def lineplot(tour):
    reporter = r0827509()
    # _, _ = reporter.optimize(tour)
    data = pd.read_csv('results_'+tour)
    sns.lineplot(data=data,x='iteration',y='fitness',hue='type')
    plt.title('Convergence plot for %s'%tour[:-4])
    plt.xticks([0,5,10,15,20,25])
    # plt.xticks([0,2,4,6,8,10,12,14,16,18,20],rotation = 45)
    plt.ylim(70000,80000)
    plt.plot()
    plt.savefig('convergence_plot_%s.png'%tour[:-4],bbox='tight',dpi=300)

    #To do plot two lines with the evolution of mean and shared fitness
    #Do not plot mean if it is infinite
if __name__=='__main__':
    # boxplot(['tour50.csv'])#,'tour100.csv','tour250.csv','tour500.csv','tour750.csv','tour1000.csv'])
    # histplot(['1-variability of results for tour50.csv','2-variability of results for tour50.csv','3-variability of results for tour50.csv',
            #   '4-variability of results for tour50.csv','5-variability of results for tour50.csv'])
    # lineplot('tour50.csv')
    # lineplot('tour100.csv')
    # lineplot('tour500.csv')
    lineplot('tour1000.csv')
    # print_minimum('tour1000.csv')