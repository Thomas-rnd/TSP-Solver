from test_algo import test_global_plus_proche_voisin, test_global_2_opt, test_global_algo_genetique, test_unitaire_2_opt

import numpy as np
import pandas as pd
from test_algo import test_unitaire_plus_proche_voisin, test_unitaire_2_opt, test_global_2_opt, test_global_plus_proche_voisin, test_unitaire_algo_genetique, test_global_algo_genetique
from graph import affichage
from init_test_data import data_TSPLIB, trajet_en_df

"""
data = data_TSPLIB('../data/dj38.tsp')
df = test_unitaire_plus_proche_voisin(0)
affichage(df, data).show()


"""
df = test_global_plus_proche_voisin()
df.to_csv('test_global_plus_proche_voisin.csv')
"""
data = data_TSPLIB('../data/qa194.tsp')
df = test_unitaire_2_opt(2)
affichage(df, data).show()

df = test_global_2_opt()
df.to_csv('test_global_2_opt.csv')

data = data_TSPLIB('../data/qa194.tsp')
df = test_unitaire_algo_genetique(1)
affichage(df, data).show()

df = test_global_algo_genetique()
df.to_csv('test_global_algo_genetique.csv')
"""
