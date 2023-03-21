import numpy as np
import pandas as pd
from scipy.spatial import distance


def distance_euclidienne(a: np.float_, b: np.float_) -> np.float_:
    """
    Retourne un np.array des distances entre 2 np.array de points.

    Parameters
    ----------
    a : np.float_
        vecteur de point 2D
    b : np.float_
        vecteur de point 2D

    Returns
    -------
    np.float_
        vecteur des distances
    """
    return np.linalg.norm(a - b, axis=1)


def matrice_distance(villes: pd.DataFrame) -> np.array:
    """
    Retourne une matrice stockant les distances inter villes. Cette matrice renseigne
    sur la distance entre la ville X et la ville Y à la position (X,Y).

    Parameters
    ----------
    villes : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir

    Returns
    -------
    Dataframe
        matrice stockant l'integralité des distances inter villes
    """
    dist_matrice = distance.cdist(
        villes[['x', 'y']], villes[['x', 'y']], 'euclidean')

    # On remplace les zéros des diagonales
    dist_matrice = np.where(dist_matrice == 0, np.Inf, dist_matrice)
    return dist_matrice


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
