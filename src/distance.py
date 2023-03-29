import numpy as np
import pandas as pd
from scipy.spatial import distance


def distance_euclidienne(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Retourne un vecteur des distances entre 2 vecteurs de points.

    Parameters
    ----------
    a : np.ndarray
        vecteur de point 2D
    b : np.ndarray
        vecteur de point 2D

    Returns
    -------
    np.ndarray
        vecteur des distances
    """
    return np.linalg.norm(a - b, axis=1)


def matrice_distance(villes: pd.DataFrame) -> np.ndarray:
    """
    Retourne une matrice stockant les distances inter villes. Cette matrice renseigne
    sur la distance entre la ville X et la ville Y à la position (X,Y).

    Parameters
    ----------
    villes : DataFrame
        dataframe stockant l'intégralité des coordonnées des villes à parcourir

    Returns
    -------
    np.ndarray
        matrice stockant l'integralité des distances inter villes
    """
    dist_matrice = distance.cdist(
        villes[['x', 'y']], villes[['x', 'y']], 'euclidean')

    # On remplace les zéros des diagonales
    dist_matrice = np.where(dist_matrice == 0, np.Inf, dist_matrice)
    return dist_matrice


def distance_trajet(itineraire: list[int], matrice_distance: np.ndarray) -> float:
    """Calcul de la distance totale d'un trajet

    Parameters
    ----------
    itineraire : list[int]
        liste ordonnées des villes parcourues
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    float
        la distance de l'itinéraire considéré
    """
    distance = 0
    for index, ville in enumerate(itineraire[:-1]):
        # distance entre la ville itineraire[index] et itineraire[index+1]
        distance += matrice_distance[ville, itineraire[index+1]]
    return distance


def neurone_gagnant(neurones: np.ndarray, ville: np.ndarray) -> np.intp:
    """On cherche le neurone le plus proche d'une ville donnée

    Parameters
    ----------
    neurones : np.ndarray
        liste du réseau de neuronnes
    ville : np.ndarray
        coordonnées 2D d'une ville donnée

    Returns
    -------
    intp
        l'index du neurone gagnant
    """
    return distance_euclidienne(neurones, ville).argmin()
