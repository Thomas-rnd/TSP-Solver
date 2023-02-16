import time

import numpy as np

from distance import distance_trajet
from graph import representation_itineraire, affichage
from testData import data_TSPLIB, tour_optimal, trajet_en_df

"""
En s'inspirant de la documentation wikipedia sur le 2-opt pour résoudre le TSP, nous
allons essayer de l'implémenter. Cet algorithme donne un résultat sub-optimal en temps
très raisonnable. L'unique optimisation que nous réaliserons concerne la maj de la distance
parcourue. En effet, on ne calculera uniquement la partie du trajet qui se verra modifiée
à la suite de l'inversion. Cela concerne nous 2 arêtes du graph à parcourir.

Cf. https://fr.wikipedia.org/wiki/2-opt
"""


def gain(matrice_distance, meilleur_chemin, i, j):
    """Gain de distance en parcourant en sens inverse une suite de ville.

    On vient calculer la différence de distance entre la somme des anciennes arêtes et
    la somme des nouvelles arêtes formées. Si cette somme est positive on vient de trouver
    deux arêtes qui étaient sécantes avant l'inversion. 

    Parameters
    ----------
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes
    meilleur_chemin : list
        suite de villes donnant le chemin parcouru
    i : int
        indice de la ville où commence l'inversion
    j : int
        indice de la ville où finie l'inversion

    Returns
    -------
    int
        Un nombre positif si l'inversion s'avère bénéfique
    """
    avant_permutation = meilleur_chemin[i-1]
    debut_permutation = meilleur_chemin[i]
    fin_permutation = meilleur_chemin[j]
    apres_permuation = meilleur_chemin[j+1]

    distance_initiale = matrice_distance[avant_permutation][debut_permutation
                                                            ]+matrice_distance[fin_permutation][apres_permuation]
    distance_finale = matrice_distance[avant_permutation][fin_permutation
                                                          ]+matrice_distance[debut_permutation][apres_permuation]
    return (distance_initiale-distance_finale)


def inversion(liste, debut_inversion, fin_inversion):
    """Inversion d'une partie du chemin parcouru

    Parameters
    ----------
    liste : list
        suite de villes donnant le chemin parcouru
    debut_inversion : int
        index de la ville où l'inversion commence
    fin_inversion : int
        index de la ville où l'inversion finie

    Returns
    -------
    int
        Un nouveau parcours avec l'inversion réalisé
    """
    nouvelle_liste = liste[:debut_inversion] + \
        list(
        reversed(liste[debut_inversion:fin_inversion+1])) + liste[fin_inversion+1:]
    return nouvelle_liste


def deux_opt(itineraire_initial, matrice_distance):
    """Recherche de deux arêtes sécantes. 

    Cette fonction implémente l'algorithme 2-opt décrit sur wikipédia.

    Parameters
    ----------
    itineraire_initial : list
        suite de villes donnant le chemin parcouru. Ce chemin initial influ énormément 
        sur la solution finale trouvée.
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    list
        Le chemin sub-optimal trouvé
    """
    start_time = time.time()

    amelioration = True
    chemin_explores = []
    # Le chemin courant est le dernier de cette liste
    chemin_explores.append(itineraire_initial)
    meilleur_distance = distance_trajet(chemin_explores[-1], matrice_distance)
    nombre_ville = len(chemin_explores[-1])

    while amelioration:
        amelioration = False
        for debut_inversion in range(1, nombre_ville - 2):
            for fin_inversion in range(debut_inversion + 1, nombre_ville - 1):
                if (gain(matrice_distance, chemin_explores[-1], debut_inversion, fin_inversion)) > 0:
                    nouveau_chemin = inversion(
                        chemin_explores[-1], debut_inversion, fin_inversion)
                    nouvelle_distance = distance_trajet(
                        nouveau_chemin, matrice_distance)

                    if (nouvelle_distance < meilleur_distance):
                        chemin_explores.append(nouveau_chemin)
                        meilleur_distance = nouvelle_distance
                        amelioration = True

    temps_calcul = time.time() - start_time
    return chemin_explores, temps_calcul


def main(matrice_distance, chemin_initial, chemin_optimal=[]):
    """Lancement de l'algorithme de recherche 

    Parameters
    ----------
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes
    chemin_initial : list
        suite de villes donnant le chemin parcouru. Ce chemin initial influ énormément 
        sur la solution finale trouvée.
    chemin_optimal : list
        résulat optimal donné par la TSPLIB

    Returns
    -------
    dict
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """

    if chemin_optimal != []:
        distance_chemin_optimal = distance_trajet(
            chemin_optimal, matrice_distance)

    resolution = {
        'Nombre de villes': len(chemin_optimal),
        'Algorithme': 'Algorithme 2-opt',
        'Distance': 'Euclidienne-2D',
        'Chemins': [],
        'Chemin optimal': chemin_optimal,
        'Erreur (en %)': 0,
        'Temps de calcul (en s)': 0
    }

    chemin_explores, temps_calcul = deux_opt(chemin_initial, matrice_distance)

    resolution['Chemins'].extend(chemin_explores)
    distance_chemin_sub_optimal = distance_trajet(
        resolution['Chemins'][-1], matrice_distance)
    if chemin_optimal != []:
        erreur = 100*(distance_chemin_sub_optimal -
                      distance_chemin_optimal)/distance_chemin_optimal
        resolution['Erreur (en %)'] = erreur

    resolution['Temps de calcul (en s)'] = temps_calcul
    return resolution
