import numpy as np
import pandas as pd


def distance_euclidienne(x1: float, y1: float, x2: float, y2: float) -> float:
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
    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def matrice_distance(data: pd.DataFrame) -> np.array:
    """Matrice des distances inter villes. C'est une matrice 2D qui renseigne
    sur la distance entre la ville X et la ville Y à la position (X,Y) de la 
    matrice

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir

    Returns
    -------
    np.array
        la matrice ainsi calculée
    """
    # Initialisation de la matrice
    distance = [[]]
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if (i == j):
                # Pas de déplacement si on reste sur la même ville
                distance[i].append(np.Inf)
            else:
                # Autrement calcul de la distance
                x1 = data.iloc[i, 1]
                y1 = data.iloc[i, 2]
                x2 = data.iloc[j, 1]
                y2 = data.iloc[j, 2]
                distance[i].append(distance_euclidienne(x1, y1, x2, y2))
        # Pour éviter d'ajouter une ligne inutilisée
        if (i != data.shape[0]-1):
            # Une fois que tous les couples de distance sont initialisés pour une ville
            # on ajoute une ville pour faire de même avec la ville suivante
            distance.append([])
    return np.array(distance)


def distance_trajet(itineraire: list, matrice_distance: np.array) -> float:
    """Calcul de la distance totale d'un trajet

    Parameters
    ----------
    itineraire : list
        Liste ordonnées des villes parcourues
    matrice_distance : np.array
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    float
        la distance de l'itinéraire considéré
    """
    distance = 0
    for i in range(len(itineraire)-1):
        # distance entre la ville itineraire[i] et itineraire[i+1]
        distance += matrice_distance[itineraire[i], itineraire[i+1]]
    return distance
