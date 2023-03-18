import random

import numpy
import pandas as pd


# Initialisation de mon dataFrame. C'ést-à-dire un ensemble de N villes definies
# par un couple de coordonnées (x,y). Les villes ont un index entre [0:N-1]
def init_random_df(n):
    """Initialisation d'un dataframe de ville à traverser

    Parameters
    ----------
    n : int
        nombre de villes présentent dans le dataframe

    Returns
    -------
    DataFrame
        L'ensemble de ville ainsi crée
    """
    # Borne des coordonnées x et y des villes
    TAILLE_FENETRE = 1000
    x = []
    y = []
    while len(x) < n:
        # Initialisation aléatoire des coordonnées d'une ville
        a = random.randint(0, TAILLE_FENETRE)
        b = random.randint(0, TAILLE_FENETRE)
        if a not in x and b not in y:
            x.append(a)
            y.append(b)
    # Initialisation du dataframe
    data = pd.DataFrame({'x': x, 'y': y})
    return (data)
