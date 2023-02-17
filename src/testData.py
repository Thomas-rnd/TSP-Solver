import numpy
import pandas as pd

# http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/


def data_TSPLIB(fichier='../data/ulysses22.txt'):
    """Lecture d'un jeu de données depuis un fichier .txt

    Initialisation des villes à traverser

    Parameters
    ----------
    fichier : string
        nom du fichier à traiter

    Returns
    -------
    DataFrame
        L'ensemble de ville ainsi crée depuis le fichier.txt
    """
    # Ouverture et lecture du fichier ligne par ligne
    with open(fichier, 'r') as input:
        data = [line.replace('\n', '') for line in input.readlines()]
    # Les coordonnées sont séparées d'un espace
    # Format des données : index_ville X Y
    coord_villes = [
        [float(i) for i in coord.split(' ')]
        for coord in data
    ]
    # Récupération des coordonnées
    X = [x[1] for x in coord_villes]
    Y = [y[2] for y in coord_villes]
    array = numpy.array([X, Y])
    data = pd.DataFrame(array, index=['x', 'y'])
    return (data)


def tour_optimal(fichier='../data/ulysses22_opt_tour.txt'):
    """Lecture d'un jeu de données depuis un fichier .txt

    Lecture du chemin optimal

    Parameters
    ----------
    fichier : string
        nom du fichier à traiter

    Returns
    -------
    list
        liste optimale du parcours des villes
    """
    # Ouverture et lecture du fichier ligne par ligne
    with open(fichier, 'r') as input:
        data = [line.replace('\n', '') for line in input.readlines()]
    # -1 pour rester en correspondance avec le dataframe ville de 0 à n-1
    tour_optimal = [int(i)-1 for i in data]
    # On revient à la ville initiale
    tour_optimal.append(tour_optimal[0])
    return tour_optimal


def trajet_en_df(trajet, data):
    """Convertion d'un trajet en un dataframe

    Parameters
    ----------
    fichier : string
        nom du fichier à traiter

    Returns
    -------
    DataFrame
        DataFrame ordonnées pour afficher le trajet
    """
    # Récupération des coordonnées des villes pour pouvoir les afficher
    x = [data.iloc[0, i] for i in trajet]
    y = [data.iloc[1, i] for i in trajet]
    array = numpy.array([x, y])
    data = pd.DataFrame(array, index=['x', 'y'])
    return (data)
