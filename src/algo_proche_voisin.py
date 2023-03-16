import time

import numpy as np
import pandas as pd

from distance import distance_trajet
from init_test_data import trajet_en_df

# En s'inspirant des cours dispensés par M. Jean-Marc Salotti en deuxième année
# à l'ENSC


def plus_proche_voisin(itineraire_initial: list, matrice_distance: pd.DataFrame) -> tuple:
    """
    Retourne le chemin parcouru en parcourant les villes de proche en proche ainsi que le 
    temps de calcul
    """
    start_time = time.time()

    # Initialiation du matrice booléenne d'état de visite des villes
    visite = np.zeros(len(itineraire_initial)) != 0

    # Initialisation de l'itinéraire
    itineraire = [0]
    visite[0] = True

    while False in visite:
        # A chaque itération on cherche la ville la plus proche de la ville actuelle
        # la ville actuelle étant la dernière de l'itinéraire
        distance_a_ville = matrice_distance.iloc[itineraire[-1], :]

        for index in range(len(distance_a_ville)):
            if visite[index]:
                distance_a_ville[index] = np.Inf

        plus_proche = np.argmin(distance_a_ville)

        visite[plus_proche] = True
        itineraire.append(plus_proche)
    # On fait attention à fermer le cycle
    itineraire.append(itineraire[0])

    temps_calcul = time.time() - start_time
    return itineraire, temps_calcul


def main(data: pd.DataFrame, matrice_distance: pd.DataFrame):
    # On récupère le chemin final trouvé ainsi que le temps de calcul
    res = plus_proche_voisin(
        data[['Ville']], matrice_distance)

    solution = res[0]
    temps_calcul = res[1]

    # Calcul de la distance du trajet final trouvé par l'algorithme. En dernière position
    # de la variable précédente
    distance_chemin_sub_optimal = distance_trajet(trajet_en_df(solution, data))

    # Création du dataframe à retourner
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': len(solution),
        # Dans un tableau pour être sur une seule ligne du dataframe
        'Solution': solution,
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Distance': distance_chemin_sub_optimal,
        'Temps de calcul (en s)': temps_calcul
    })

    return df_resultat_test