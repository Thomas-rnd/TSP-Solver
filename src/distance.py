import numpy


def distance_euclidienne(x1, y1, x2, y2):
    """Evaluation de la distance euclidienne entre 2 points en 2D

    Parameters
    ----------
    x1 : int
        coordonné x du point 1
    x2 : int
        coordonné x du point 2
    y1 : int
        coordonné y du point 1
    y2 : int
        coordonné y du point 2

    Returns
    ----------
    int
        la distance calculée
    """
    distance = numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def matrice_distance(data):
    """Matrice des distances inter villes. C'est une matrice 2D qui renseigne
    sur la distance entre la ville X et la ville Y à la position (X,Y) de la 
    matrice

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir

    Returns
    -------
    list
        la matrice ainsi calculée
    """
    # Initialisation de la matrice
    distance = [[]]
    for i in range(len(data.loc['x'])):
        for j in range(len(data.loc['x'])):
            if (i == j):
                # En diagonale on a uniquement des 0. Pas de déplacement si on reste
                # sur la même ville
                distance[i].append(0)
            else:
                # Autrement calcul de la distance
                x1 = data.iloc[:, i].x
                y1 = data.iloc[:, i].y
                x2 = data.iloc[:, j].x
                y2 = data.iloc[:, j].y
                distance[i].append(distance_euclidienne(x1, y1, x2, y2))
        # Pour éviter d'ajouter une ligne inutilisée
        if (i != len(data.loc['x'])-1):
            # Une fois que tous les couples de distance sont initialisés pour une ville
            # on ajoute une ville pour faire de même avec la ville suivante
            distance.append([])
    return distance


def distance_trajet(itineraire, matrice_distance):
    """Evaluation des trajets en fonction de leur distance totale

    Parameters
    ----------
    itineraire : list
        Liste ordonnées des villes parcourues
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    int
        la distance de l'itinéraire considéré
    """
    distance = 0
    for i in range(len(itineraire)-1):
        # distance entre la ville itineraire[i] et itineraire[i+1]
        distance += matrice_distance[itineraire[i]][itineraire[i+1]]
    return distance
