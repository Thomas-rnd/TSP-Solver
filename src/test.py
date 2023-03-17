import numpy as np
import pandas as pd
from scipy.spatial import distance
from test_algo import test_unitaire_plus_proche_voisin, test_unitaire_2_opt
from graph import representation_itineraire_web, representation_itineraire_back
from init_test_data import data_TSPLIB, trajet_en_df

"""
data = data_TSPLIB('../data/qa194.tsp')
df = test_unitaire_plus_proche_voisin(0)
df_solution = trajet_en_df(df['Solution'], data)
representation_itineraire_web(df_solution).show()
"""

data = data_TSPLIB('../data/qa194.tsp')
df = test_unitaire_2_opt(1)
df_solution = trajet_en_df(df['Solution'][0], data)
print(df)
representation_itineraire_web(df_solution).show()
