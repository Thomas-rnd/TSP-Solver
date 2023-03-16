import numpy as np
import pandas as pd
from scipy.spatial import distance


def distance_euclidienne(a: np.float_, b: np.float_) -> np.float_:
    """
    Retourne un np.array des distances entre 2 np.array de points.
    """
    return np.linalg.norm(a - b, axis=1)


def matrice_distance(villes: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un dataframe stockant les distances inter villes. C'est un dataframe qui renseigne
    sur la distance entre la ville X et la ville Y à la position (X,Y).
    """
    dist_matrice = distance.cdist(
        villes[['x', 'y']], villes[['x', 'y']], 'euclidean')

    # On remplace les zéros des diagonales
    dist_matrice = np.where(dist_matrice == 0, np.Inf, dist_matrice)
    df = pd.DataFrame(dist_matrice, index=villes[[
                      'Ville']], columns=villes[['Ville']])
    return df


def distance_trajet(villes: pd.DataFrame) -> float:
    """
    Retourne la distance pour traverser un chemin de villes dans un certain ordre
    """
    points = villes[['x', 'y']]
    distances = distance_euclidienne(points, np.roll(points, 1, axis=0))

    return np.sum(distances)
