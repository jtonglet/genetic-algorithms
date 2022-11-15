
#Compute variation in values with a boxplot on tour 50 (100 iterations)
from r0123456 import r0123456, Parameters
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

repetitions = 50
scores = []
init = []
tour = 'tour50.csv'
params = Parameters(400,100,2,500,0.8)
for i in range(repetitions):
    reporter = r0123456()
    scores.append(reporter.optimize(tour,params))
    init.append('E-Greedy')
for i in range(repetitions):
    reporter = r0123456()
    scores.append(reporter.optimize(tour,params,heuristic=False))
    init.append('Random')

data = pd.DataFrame({'fitness':scores,
                     'initialization':init})

sns.boxplot(x=data['fitness'],y=data['initialization'])
plt.title('Variability of the best fitness for tour 50')
plt.xticks(rotation = 45)
plt.plot()
plt.savefig('boxplot.png',bbox='tight',dpi=300)