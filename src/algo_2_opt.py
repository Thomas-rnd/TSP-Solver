import time

import numpy as np
import pandas as pd

from distance import distance_trajet
from init_test_data import trajet_en_df

"""
En s'inspirant de la documentation wikipedia sur le 2-opt pour résoudre le TSP, nous
allons essayer de l'implémenter. Cet algorithme donne un résultat sub-optimal en temps
très raisonnable. L'unique optimisation que nous réaliserons concerne l'évaluation des
distances. En effet, on ne calculera uniquement la partie du trajet qui se verra modifiée
à la suite de l'inversion, c'est-à-dire 2 arêtes.

Cf. https://fr.wikipedia.org/wiki/2-opt
"""


def gain(matrice_distance: pd.DataFrame, meilleur_chemin: list, i: int, j: int) -> int:
    """Gain de distance en parcourant en sens inverse une suite de ville.

    On vient calculer la différence de distance entre la somme des anciennes arêtes et
    la somme des nouvelles arêtes formées. Si cette somme est positive on vient de trouver
    deux arêtes qui étaient sécantes avant l'inversion.

    Parameters
    ----------
    matrice_distance : Dataframe
        matrice stockant l'integralité des distances inter villes
    meilleur_chemin : list
        suite de villes donnant le chemin parcouru
    i : int
        indice de la ville où commence l'inversion
    j : int
        indice de la ville où finie l'inversion

    Returns
    -------
    bool
       True si l'inversion s'avère bénéfique
    """
    avant_permutation = meilleur_chemin[i-1]
    debut_permutation = meilleur_chemin[i]
    fin_permutation = meilleur_chemin[j]
    apres_permuation = meilleur_chemin[j+1]

    # Calcul des distances ce voyant modifiées avec l'inversion
    # Les 2 anciennes arêtes
    distance_initiale = matrice_distance.iloc[avant_permutation,
                                              debut_permutation] + matrice_distance.iloc[fin_permutation, apres_permuation]
    # Les 2 nouvelles arêtes
    distance_finale = matrice_distance.iloc[avant_permutation, fin_permutation
                                            ] + matrice_distance.iloc[debut_permutation, apres_permuation]
    return (distance_initiale-distance_finale)


def inversion(chemin_courant: list, debut_inversion: int, fin_inversion: int) -> list:
    """Inversion d'une partie du chemin parcouru

    Parameters
    ----------
    chemin_courant : list
        suite de villes donnant le chemin parcouru
    debut_inversion : int
        index de la ville où l'inversion commence
    fin_inversion : int
        index de la ville où l'inversion finie

    Returns
    -------
    list
        Un nouveau parcours avec l'inversion réalisée
    """
    nouvelle_liste = chemin_courant[:debut_inversion] + \
        list(
        reversed(chemin_courant[debut_inversion:fin_inversion+1])) + chemin_courant[fin_inversion+1:]
    return nouvelle_liste


def deux_opt(data: pd.DataFrame, itineraire_initial: list, matrice_distance: pd.DataFrame):
    """Recherche de deux arêtes sécantes.

    Cette fonction implémente l'algorithme 2-opt décrit sur wikipédia.

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    itineraire_initial : list
        suite de villes donnant le chemin parcouru. Ce chemin initial influ énormément
        sur la solution finale trouvée.
    matrice_distance : DataFrame
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    chemin_explores : list
        L'ensemble des chemins sub-optimal trouvés
    temps_calcul : int
        temps necessaire à la résolution du problème
    """
    start_time = time.time()

    amelioration = True
    meilleur_chemin = itineraire_initial
    meilleur_distance = distance_trajet(trajet_en_df(meilleur_chemin, data))
    nombre_ville = len(meilleur_chemin)

    while amelioration:
        amelioration = False
        # Parcours de l'ensemble des villes en réalisant l'ensemble des permutations possibles
        for debut_inversion in range(1, nombre_ville - 2):
            for fin_inversion in range(debut_inversion + 1, nombre_ville - 1):
                gain_trajet = gain(
                    matrice_distance, meilleur_chemin, debut_inversion, fin_inversion)
                if (gain_trajet > 0):
                    nouveau_chemin = inversion(
                        meilleur_chemin, debut_inversion, fin_inversion)
                    nouvelle_distance = meilleur_distance-gain_trajet
                    # On regarde si le chemin est meilleur que le chemin courant
                    if (nouvelle_distance < meilleur_distance):
                        meilleur_chemin = nouveau_chemin
                        meilleur_distance = nouvelle_distance
                        amelioration = True

    temps_calcul = time.time() - start_time
    return meilleur_chemin, temps_calcul


def main(data: pd.DataFrame, matrice_distance: pd.DataFrame, chemin_initial: list) -> pd.DataFrame:
    """Lancement de l'algorithme de recherche sur 1 jeu de données

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : DataFrame
        matrice stockant l'integralité des distances inter villes
    chemin_initial : list
        suite de villes donnant le chemin parcouru. Ce chemin initial influ énormément
        sur la solution finale trouvée.

    Returns
    -------
    Dataframe
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme
    """

    # On récupère les chemins testés et le temps de résolution de l'algorithme
    itineraire, temps_calcul = deux_opt(data, chemin_initial, matrice_distance)

    # Calcul de la distance du trajet final trouvé par l'algorithme. En dernière position
    # de la variable précédente
    distance_chemin_sub_optimal = distance_trajet(
        trajet_en_df(itineraire, data))

    # Création du dataframe à retourner
    df_resultat_test = pd.DataFrame({
        'Algorithme': "2-opt",
        'Nombre de villes': len(chemin_initial),
        # Dans un tableau pour être sur une seule ligne du dataframe
        'Solution': [itineraire],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Distance': distance_chemin_sub_optimal,
        'Temps de calcul (en s)': temps_calcul
    })

    return df_resultat_test
