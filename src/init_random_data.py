import numpy as np
import pandas as pd


def init_random_df(n: int) -> pd.DataFrame:
    """
    Initialisation d'un dataframe de ville à traverser
    """
    # Borne des coordonnées x et y des villes
    TAILLE_FENETRE = 1000
    x = np.random.random_sample(n)*TAILLE_FENETRE
    y = np.random.random_sample(n)*TAILLE_FENETRE
    index = np.arange(n, dtype=int)

    # Initialisation du dataframe
    data = pd.DataFrame({'Ville': index, 'x': x, 'y': y})
    return (data)
