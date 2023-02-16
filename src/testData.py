import numpy
import pandas as pd

# http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/


def data_TSPLIB(fichier='../data/ulysses22.txt'):
    with open(fichier, 'r') as input:
        data = [line.replace('\n', '') for line in input.readlines()]
    coord_villes = [
        [float(i) for i in coord.split(' ')]
        for coord in data
    ]
    X = [x[1] for x in coord_villes]
    Y = [y[2] for y in coord_villes]
    array = numpy.array([X, Y])
    data = pd.DataFrame(array, index=['x', 'y'])
    return (data)


def tour_optimal(fichier='../data/ulysses22_opt_tour.txt'):
    with open(fichier, 'r') as input:
        data = [line.replace('\n', '') for line in input.readlines()]
    # -1 pour rester en correspondance avec le dataframe ville de 0 à n-1
    tour_optimal = [int(i)-1 for i in data]
    return tour_optimal


def trajet_en_df(trajet, data):
    """Convertion d'un trajet en un dataframe"""
    x = [data.iloc[0, i] for i in trajet]
    y = [data.iloc[1, i] for i in trajet]
    array = numpy.array([x, y])
    data = pd.DataFrame(array, index=['x', 'y'])
    return (data)
