import itertools
import random
import time

import numpy as np
from distance import distance_trajet
from graph import representation_itineraire
from testData import data_TSPLIB, tour_optimal, trajet_en_df


def gain(matrice_distance, meilleur_chemin, i, j):
    """Calcul de la distance parcouru uniquement sur la modification de l'itinéraire. La modification 
    de l'itinéraire est située au niveau de la cassure des deux arêtes choisies pour la permutation. 
    Cf. https://fr.wikipedia.org/wiki/2-opt . Si le gain est positif alors nouveau trajet est plus court."""
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
    nouvelle_liste = liste[:debut_inversion] + \
        list(
        reversed(liste[debut_inversion:fin_inversion+1])) + liste[fin_inversion+1:]
    return nouvelle_liste


def deux_opt(itineraire_initial, matrice_distance, data):
    amelioration = True
    meilleur_chemin = itineraire_initial
    meilleur_distance = distance_trajet(meilleur_chemin, matrice_distance)
    nombre_ville = len(meilleur_chemin)

    while amelioration:
        amelioration = False
        for debut_inversion in range(1, nombre_ville - 2):
            for fin_inversion in range(debut_inversion + 1, nombre_ville - 1):
                if (gain(matrice_distance, meilleur_chemin, debut_inversion, fin_inversion)) > 0:
                    nouveau_chemin = inversion(
                        meilleur_chemin, debut_inversion, fin_inversion)
                    nouvelle_distance = distance_trajet(
                        nouveau_chemin, matrice_distance)

                    if (nouvelle_distance < meilleur_distance):
                        meilleur_chemin = nouveau_chemin
                        # representationParcours(trajetToDataframe(meilleurChemin, data))
                        meilleur_distance = nouvelle_distance
                        amelioration = True
    return meilleur_chemin


def affichage(resolution, data):
    # Affichage pour observer ou non la convergence
    df_meilleur_trajet = trajet_en_df(resolution['Chemins'][-1], data)
    representation_itineraire(df_meilleur_trajet)


def main(data, matrice_distance, chemin_initial, chemin=[]):
    start = time.time()

    if chemin != []:
        distance_chemin_optimal = distance_trajet(chemin, matrice_distance)

    resolution = {
        'Nombre de villes': len(chemin),
        'Algorithme': 'Algorithme 2-opt',
        'Distance': 'Euclidienne-2D',
        'Chemins': [],
        'Chemin optimal': chemin,
        'Erreur (en %)': [],
        'Temps de calcul': 0
    }
    trajet = {'Villes': [], 'Distance': 0}
    trajet['Villes'] = chemin_initial
    # Le marchand revient sur ses pas
    trajet['Distance'] = distance_trajet(trajet['Villes'], matrice_distance)

    res = {'Villes': [], 'Distance': 0}
    res['Villes'] = deux_opt(trajet['Villes'], matrice_distance, data)
    res['Distance'] = distance_trajet(trajet['Villes'], matrice_distance)

    resolution['Chemins'].append(res['Villes'])
    if chemin != []:
        erreur = 100*(res['Distance'] -
                      distance_chemin_optimal)/distance_chemin_optimal
        resolution['Erreur (en %)'].append(erreur)

    resolution['Temps de calcul'] = time.ctime(time.time() - start)
    return resolution
