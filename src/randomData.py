import random

import numpy
import pandas as pd


# Initialisation de mon dataFrame. C'ést-à-dire un ensemble de N villes definies
# par un couple de coordonnées (x,y). Les villes ont un index entre [0:N-1]
def init_random_df(n):
    taille_fenetre = 1000
    x = []
    y = []
    while len(x) < n:
        a = random.randint(0, taille_fenetre)
        b = random.randint(0, taille_fenetre)
        if a not in x and b not in y:
            x.append(a)
            y.append(b)
    array = numpy.array([x, y])
    data = pd.DataFrame(array, index=['x', 'y'])
    return (data)
