import pandas as pd
import numpy as np
import time
import os

data_dir = '/Users/michael/Documents/github/Nifty-Fifty/Kaggle_Christmas/Data'
data = pd.read_csv(data_dir+'/family_data.csv', index_col='family_id')
submission = pd.read_csv(data_dir+'/sample_submission.csv', index_col='family_id')


N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125
INDEX = np.linspace(0,4999,5000, dtype=int)
GIFT  = np.asarray([0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500])
DISCOUNT = np.asarray([0, 0, 9, 9, 9, 18, 18, 36, 36, 235, 434])
CHOICES = np.asarray(data.drop('n_people', 1))
PEOPLE = np.asarray(data['n_people'])




def loss(prediction):

    occupency = np.zeros((5000,100))
    occupency[INDEX, prediction - 1] = PEOPLE
    occupency = np.sum(occupency,0)
    occupency = occupency.astype(int)

    if np.sum(occupency>MAX_OCCUPANCY)+np.sum(occupency<MIN_OCCUPANCY):
        return 100000000

    prediction = np.tile(prediction, (10, 1))
    prediction = np.swapaxes(prediction, 0,1)

    success = (CHOICES == prediction)
    failed = np.sum(success, axis=1)*(-1)+1
    day = np.c_[success, failed]

    cost = np.sum(np.matmul(day,GIFT) + np.matmul(day, DISCOUNT)*PEOPLE)

    diff = 0.5+np.abs(np.append(np.diff(occupency),0))/50
    cost+= np.sum(np.power(occupency,diff)*(occupency - 125)/400)

    return cost




SIZE = 5000
N_POPULATION = 1000
MATING_NUM = 10
N_CHILDREN = 30
TOT_CHILDREN = MATING_NUM * N_CHILDREN
MUTATION_PROB = 0.001
EPOCHS = 3000
KEEP_BEST = 100

population = np.random.randint(100, size=(SIZE,N_POPULATION))
# for i in range(N_POPULATION):
#     population[:,i] = CHOICES[INDEX,np.random.randint(10,size=(SIZE))]

load = False
if load:
    population = np.load('best_pop.npy')


new_pop = np.zeros((SIZE, MATING_NUM*N_CHILDREN), dtype=np.uint8)
loss_history = []

for epoch in range(EPOCHS):

    fitness = np.zeros(N_POPULATION)
    for i in range(N_POPULATION):
        fitness[i] = loss(population[:,i])

    population_ranking = np.argsort(fitness)
    population = population[:,population_ranking]

    print(epoch, int(fitness[population_ranking[0]]), int(fitness[population_ranking[1]]))
    loss_history.append(fitness[population_ranking[0]])


    k = 0
    for i in range(0, MATING_NUM, 2):
        # parent_1 = np.unpackbits(population[:,i].astype(np.uint8))
        # parent_2 = np.unpackbits(population[:,i+1].astype(np.uint8))

        parent_1 = population[:,i]
        parent_2 = population[:,i+1]

        for j in range(N_CHILDREN):

            ## CROSSOVER
            x_over_pt = np.random.randint(SIZE)
            child = np.append(parent_1[:x_over_pt], parent_2[x_over_pt:])

            ## MUTATION
            pts = np.random.randint(SIZE, size = int(SIZE * MUTATION_PROB))

            child[pts] = CHOICES[pts,np.random.randint(10, size = int(SIZE * MUTATION_PROB))]
            # np.random.randint(100, size=int(SIZE * MUTATION_PROB))

            new_pop[:,k] = child
            k = k+1

    population[:, KEEP_BEST : KEEP_BEST + TOT_CHILDREN] = new_pop.astype(int)
    population[:, KEEP_BEST + TOT_CHILDREN :] = np.random.randint(100, size=(SIZE, N_POPULATION - KEEP_BEST - TOT_CHILDREN))

np.save('best_pop2.npy', population)

import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.show()



