import numpy


def distance_euclidienne(x1, y1, x2, y2):
    """Evaluation de la distance entre 2 points en 2D"""
    distance = numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def matrice_distance(data):
    distance = [[]]
    for i in range(len(data.loc['x'])):
        for j in range(len(data.loc['x'])):
            if (i == j):
                distance[i].append(0)
            else:
                x1 = data.iloc[:, i].x
                y1 = data.iloc[:, i].y
                x2 = data.iloc[:, j].x
                y2 = data.iloc[:, j].y
                distance[i].append(distance_euclidienne(x1, y1, x2, y2))
        # Pour éviter d'ajouter une ligne inutilisée
        if (i != len(data.loc['x'])-1):
            distance.append([])
    return distance


def distance_trajet(itineraire, matrice_distance):
    """Evaluation des trajets en fonction de leur distance totale"""
    distance = 0
    for i in range(len(itineraire)-1):
        distance += matrice_distance[itineraire[i]][itineraire[i+1]]
    return distance
