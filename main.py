
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as random
import copy
import operator

citys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
crossoverpoint = 8
crossoverpoint2 = 18

numberOfindividuals = 100

tour = int(numberOfindividuals * 0.08)
generations = 1000

population = []
#matriz adjacency
matrix = [
[0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6], 
[4, 2, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 4, 0, 0, 5, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0], 
[0, 6, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 3, 0], 
[0, 0, 0, 0, 0, 3, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0], 
[0, 0, 12, 0, 0, 7, 4, 0, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 7, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 8, 7], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 6, 0, 0, 5, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 1, 0, 0, 0, 0, 0, 12, 6, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 7, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
] 

def punishment(arrey):
    for item in range(len(arrey)-1):
        for item2 in range(len(arrey)-1):
            if(item != item2 and arrey[item][item2]==0):
                arrey[item][item2] =200
    return arrey


def calc_dist(pop,matrix):
    dist = 0
    for item in range(len(pop)-1):
        dist = dist + matrix[pop[item]][pop[item+1]]
    return dist

def generateInitialPop(citys,matrix_int):
    for ind in range(numberOfindividuals):
        newCitys = random.sample(citys, len(citys))
        vet = newCitys
        newCitysWithFitness = (calc_dist(newCitys,matrix_int))
        vet.append(newCitysWithFitness)
        population.append(vet)
    return population


def selection(population):
    initTour = random.sample(range(0,numberOfindividuals),tour)
    L =[]
    for item in initTour:
        L.append(copy.deepcopy(population[item]))
    L = sorted(L, key=operator.itemgetter(len(citys)))
    return L[0:2]


def mediaFitness(population):
    value = 0
    for i in population:
        value = value + i[len(i)-1]
    return value/len(population)

def bestFitness(population):
    x = sorted(copy.deepcopy(population),key=operator.itemgetter(len(citys)))
    return x[0][len(citys)]


def pmx(arrey1, arrey2):
    son = []
    count = 0
    for i in arrey1:
        if(count == crossoverpoint):
            break
    if(i not in arrey2[crossoverpoint:crossoverpoint2]):
        son.append(i)
        count = count+1
    son.extend(arrey2[crossoverpoint:crossoverpoint2])
    son.extend([x for x in arrey1 if x not in son])
    return son



def mutation (population,randGene1,randGene2):
    gene1 = population[randGene1]
    gene2 = population[randGene2]
    population[randGene1] = gene2
    population[randGene2] = gene1
    return population



matrix = punishment(matrix)
pop = citys

popInt = generateInitialPop(pop,matrix)

newPop = []
graf = []

for geracao in range(1000):

    for item in range(5):
        estorcatico = selection(popInt)
        estorcatico[0].pop(len(citys))
        estorcatico[1].pop(len(citys))
        pmxV = pmx(estorcatico[0],estorcatico[1])
        mutacaoX = random.sample(range(1, 100),1)
        if mutacaoX[0] < 10:
            pmxV = mutation(pmxV,5,10)
        pmxV.append(calc_dist(pmxV,matrix))
        newPop.append(pmxV)
    for app in range(numberOfindividuals - 5):
        newPop.append(popInt[app])

    popInt = copy.deepcopy(newPop)
    newPop = []
    graf.append([geracao,mediaFitness(popInt),bestFitness(popInt)])
    print(f'Fim geração: {geracao} média: {mediaFitness(popInt)} : best: {bestFitness(popInt)}')
