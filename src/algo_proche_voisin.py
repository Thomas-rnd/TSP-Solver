import time

import numpy as np
import pandas as pd

from distance import distance_trajet
from init_test_data import trajet_en_df

# En s'inspirant des cours dispensés par M. Jean-Marc Salotti en deuxième année
# à l'ENSC


def plus_proche_voisin(matrice_distance: pd.DataFrame):
    """
    Retourne le chemin parcouru en parcourant les villes de proche en proche ainsi que le 
    temps de calcul

    Parameters
    ----------
    matrice_distance : DataFrame
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    itineraire : list
        suite de villes donnant le chemin parcouru
    temps_calcul : float
        temps de calcul pour réaliser la recherche
    """
    start_time = time.time()

    # Initialiation du matrice booléenne d'état de visite des villes
    visite = np.zeros(matrice_distance.shape[0]) != 0

    # Initialisation de l'itinéraire
    itineraire = [0]
    visite[0] = True

    while False in visite:
        # A chaque itération on cherche la ville la plus proche de la ville actuelle
        # la ville actuelle étant la dernière de l'itinéraire

        # Pour ne pas modifier le paramètre matrice_distance
        distance_a_ville = matrice_distance.iloc[itineraire[-1], :].copy()

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
    """
    Lancement de l'algorithme de recherche sur 1 jeu de données

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : DataFrame
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    Dataframe
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme
    """

    # On récupère le chemin final trouvé ainsi que le temps de calcul
    itineraire, temps_calcul = plus_proche_voisin(matrice_distance)

    # Calcul de la distance du trajet final trouvé par l'algorithme. En dernière position
    # de la variable précédente
    distance_chemin_sub_optimal = distance_trajet(
        trajet_en_df(itineraire, data))

    # Création du dataframe à retourner
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': len(itineraire),
        # Dans un tableau pour être sur une seule ligne du dataframe
        'Solution': [itineraire],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Distance': distance_chemin_sub_optimal,
        'Temps de calcul (en s)': temps_calcul
    })

    return df_resultat_test
