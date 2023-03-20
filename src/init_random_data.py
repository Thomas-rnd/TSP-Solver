import random

import pandas as pd


def init_random_df(n: int) -> pd.DataFrame:
    """Initialisation d'un dataframe de ville à traverser

    Un ensemble de N villes definies par un couple de coordonnées (x,y). 
    Les villes ont un index entre [0:N-1]

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
        # Ajout si la ville n'est pas déjà présente
        if a not in x and b not in y:
            x.append(a)
            y.append(b)
    index = [a for a in range(len(x))]
    # Initialisation du dataframe
    data = pd.DataFrame({'Ville': index, 'x': x, 'y': y})
    return (data)
