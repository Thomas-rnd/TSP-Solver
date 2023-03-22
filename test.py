from src.test_algo import test_unitaire_plus_proche_voisin, test_unitaire_2_opt, test_global_2_opt, test_global_plus_proche_voisin, test_unitaire_algo_genetique, test_unitaire_algo_kohonen, test_global_algo_genetique
from src.affichage_resultats import affichage, representation_temps_calcul, representation_resultats
from src.init_test_data import data_TSPLIB
import pandas as pd

"""
# Création d'un dataframe de résultat de test par algo

# Algo plus proche voisin
df_1 = test_global_plus_proche_voisin()
df_1.to_csv('../resultats/csv/test_global_plus_proche_voisin.csv')

# Algo 2-opt
df_2 = test_global_2_opt()
df_2.to_csv('../resultats/csv/test_global_2_opt.csv')

# Algo génétique
df_3 = test_global_algo_genetique()
df_3.to_csv('../resultats/csv/test_global_algo_genetique.csv')

# Concaténation dans un seul fichier
df = pd.concat([df_1, df_2], ignore_index=True)
df.to_csv('../resultats/csv/test_global_algos.csv')

# Test unitaire sur un jeu de données

# Algo plus proche voisin
data = data_TSPLIB('../data/dj38.tsp')
df = test_unitaire_plus_proche_voisin(0)
affichage(df, data).show()

# Algo 2-opt
data = data_TSPLIB('../data/dj38.tsp')
df = test_unitaire_2_opt(0)
affichage(df, data).show()

# Algo génétique
data = data_TSPLIB('../data/dj38.tsp')
df = test_unitaire_algo_genetique(0)
affichage(df, data).show()


# Création du visuel de visualisation des distances
representation_resultats('resultats/csv/test_global_algos.csv')
representation_temps_calcul('resultats/csv/test_global_algos.csv')
"""

data = data_TSPLIB('data/dj38.tsp')
df = test_unitaire_algo_kohonen(0)
print(df)
affichage(df, data).show()
