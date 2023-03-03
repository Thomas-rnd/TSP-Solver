import copy
import random
import time

import matplotlib.pyplot as plt
import numpy
import pandas as pd

from distance import distance_trajet, matrice_distance
from graph import affichage
from testData import data_TSPLIB, tour_optimal, trajet_en_df

# Taille de la population initiale
NOMBRE_TRAJET = 1000

# Pourcentage de population conservé à chaque épisode
POURCENTAGE_SELECTION = 20/100

# Pourcentage de mutation
POURCENTAGE_MUTATION = 25/100

# Constante permettant d'arrêter la convergence de l'algorithme
ERREUR_SUR_CHEMIN = 25


def init_population(nombre_de_trajet, data, matrice_distance):
    """Initialisation de la population initiale 

    Construction d'une population initiale de N solutions.
    Une solution étant un trajet passant par toutes les villes une et une seule fois

    Parameters
    ----------
    nombre_de_trajet : int
        taille de la population initiale
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    list
        L'ensemble des N trajets distincts crées
    """
    # Liste des solutions
    trajets = []
    for i in range(nombre_de_trajet):
        # Initialisation de la forme d'une solution
        # On instancie un nouveau dictionnaire à chaque foid
        # car type référence
        trajet = {"Villes": [], 'Distance': 0}
        # Génération d'un ordre de parcours des villes de manière aléatoire
        trajet['Villes'] = random.sample(
            range(0, len(data.loc['x'])), len(data.loc['x']))
        # Le marchand revient sur ses pas
        trajet['Villes'].append(trajet['Villes'][0])
        # Calcul de la distance total du parcours
        trajet = evaluation(trajet, matrice_distance)
        trajets.append(trajet)
    return trajets


def individus_ordonnes(trajets):
    """Tri des trajts par ordre croissant de leur distance 

    Parameters
    ----------
    trajet : dict
        ordre de parcours des villes et distance du trajet

    Returns
    -------
    list
        l'ensemble des N trajets triés
    """
    return sorted(trajets, key=lambda x: x['Distance'])


def selection(trajets, pourcentage):
    """Sélection des n% meilleurs

    Parmi la population totale on ne conserve qu'un petit pourcentage
    des meilleurs

    Parameters
    ----------
    trajet : dict
        ordre de parcours des villes et distance du trajet
    pourcentage : int
        le pourcentage à garder de la population initiale

    Returns
    -------
    list
        les n% meilleurs de la population initiale
    """
    # Nombre de trajet après sélection
    nombre_selectionne = int(len(trajets)*pourcentage)
    # Récupération de ces n trajets
    trajets = trajets[:nombre_selectionne]
    return trajets


def probabilite(pourcentage):
    """Tirage aléatoire simulant une probabilité de succés

    Parameters
    ----------
    pourcentage : int
        le pourcentage de succés

    Returns
    -------
    bool
        Succés ou non succés
    """
    r = random.randint(0, 100)
    if r <= pourcentage:
        return True
    return False


"""
Pour les mutations il est important de conserver l'intégrité de nos trajets. Le point initial est confondu
avec le point final. Le choix du point initial est arbitraire, il n'éxiste pas une première ville meilleur que les autres. 
Prenant en compte ce principe, on effectura pas de permutation affectant les extrémités du trajet.
"""


def cadre_mutation(nbVilles):
    """Définition des villes pouvant permuter en fonction de la remarque précédente.

    Les villes qui peuvent permutter sont comprise entre la première et l'avant avant dernière
    On permutte avec la ville suivante donc l'avant dernière permuterait avec la dernière ce qui 
    n'est pas possible

    Parameters
    ----------
    nbVilles : int
        nombre de ville à traverser 

    Returns
    -------
    list
        index des villes qui peuvent muter
    """
    # Initialisation du cadre
    villesMutables = [1, nbVilles-3]
    return villesMutables


def mutation_sucessive(trajet):
    """Définition d'une mutation d'un individu

    Cette mutation est une permutation aléatoire de deux villes successives

    Parameters
    ----------
    trajet : dict
        ordre de parcours des villes et distance du trajet

    Returns
    -------
    list
        nouvel ordre de parcours des villes après mutation
    """
    # Création d'un nouveau dictionnaire pour stocker le trajet muté
    enfant = {'Villes': [], 'Distance': 0}
    villes_mutables = cadre_mutation(len(trajet['Villes']))
    # Je ne veux pas modifier le trajet initial. Comme c'est un type référence
    # je réalise une copie particulière pour ne pas pointer vers la même adresse mémoire
    enfant['Villes'] = copy.deepcopy(trajet['Villes'])
    NOMBRE_PERMUTATIONS = 5
    for i in range(NOMBRE_PERMUTATIONS):
        # Indice de l'élément à permuter avec le suivant
        r = random.randint(villes_mutables[0], villes_mutables[1])
        # Permutation des deux éléments
        enfant['Villes'][r], enfant['Villes'][r +
                                              1] = enfant['Villes'][r+1], enfant['Villes'][r]
    return (enfant)


def mutation_aleatoire(trajet):
    """Définition d'une mutation d'un individu

    Cette mutation est une permutation aléatoire de deux villes

    Parameters
    ----------
    trajet : dict
        ordre de parcours des villes et distance du trajet

    Returns
    -------
    list
        nouvel ordre de parcours des villes après mutation
    """
    # Création d'un nouveau dictionnaire pour stocker le trajet muté
    enfant = {'Villes': [], 'Distance': 0}
    villes_mutables = cadre_mutation(len(trajet['Villes']))
    # Je ne veux pas modifier le trajet initial. Comme c'est un type référence
    # je réalise une copie particulière pour ne pas pointer vers la même adresse mémoire
    enfant['Villes'] = copy.deepcopy(trajet['Villes'])
    NOMBRE_PERMUTATIONS = 5
    for i in range(NOMBRE_PERMUTATIONS):
        # Indice des éléments à permuter cette fois si ils ne sont pas forcément l'un après l'autre
        r = random.sample(range(villes_mutables[0], villes_mutables[1]), 2)
        # Permutation des deux éléments
        enfant['Villes'][r[0]], enfant['Villes'][r[1]
                                                 ] = enfant['Villes'][r[1]], enfant['Villes'][r[0]]
    return (enfant)


def generation(trajets_originels, nombre_de_trajet, pourcentage_mutation, matrice_distance):
    """Génération d'une nouvelle population de N trajets

    Generation de m nouveaux trajets pour compléter la population sélectionnée. Ces nouveaux
    trajets sont des enfants de ces trajets

    Parameters
    ----------
    trajets_originels : list
        ensemble des trajets sélectionnés
    nombre_de_trajet : int
        taille de la population initiale
    pourcentage_mutation : int
        probabilité qu'un trajet mute
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    list
        L'ensemble des N trajets distincts crées
    """
    while len(trajets_originels) < nombre_de_trajet:
        for trajet in trajets_originels:
            # Génération des altérations : mutation, croisement, ...
            if probabilite(pourcentage_mutation):
                trajet = mutation_sucessive(trajet)
                trajet = evaluation(trajet, matrice_distance)
                trajets_originels.append(trajet)
            elif probabilite(pourcentage_mutation):
                trajet = mutation_aleatoire(trajet)
                trajet = evaluation(trajet, matrice_distance)
                trajets_originels.append(trajet)
    return trajets_originels


def evaluation(trajet, matrice_distance):
    """Fonction d'évalusation de l'algorithme

    Evaluation de la population. Plus un trajet est court plus il est considéré comme bon

    Parameters
    ----------
    trajet : dict
        ordre de parcours des villes avec distance relative
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    list
        un trajet avec distance mise à jour
    """
    maj_trajet = copy.deepcopy(trajet)
    maj_trajet['Distance'] = distance_trajet(
        trajet['Villes'], matrice_distance)
    return maj_trajet


def main(data, matrice_distance, chemin_optimal):
    """Lancement de l'algorithme de recherche 

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes
    chemin_optimal : list
        résulat optimal donné par la TSPLIB

    Returns
    -------
    dict
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """
    # Distance chemin optimal
    distance_chemin_optimal = distance_trajet(chemin_optimal, matrice_distance)

    # Initialisation de n individus initiaux (Génèse)
    trajets_initiaux = init_population(NOMBRE_TRAJET, data, matrice_distance)

    # Formalisme de stockage des résulats de l'algorithme
    resolution = {
        'Nombre de villes': len(chemin_optimal),
        'Algorithme': 'Algorithme génétique',
        'Distance': 'Euclidienne-2D',
        'Chemins': [],
        'Chemin optimal': chemin_optimal,
        'Erreur (en %)': [100],
        'Temps de calcul (en s)': 0
    }

    # Evaluation du temps de calcul
    start = time.time()
    # On arrète l'algorithme quand une approximation du chemin optimal est atteinte
    while resolution['Erreur (en %)'][-1] > ERREUR_SUR_CHEMIN:
        # Tri
        trajets_ordonnes = individus_ordonnes(trajets_initiaux)

        # Ajout du meilleur trajet au résultat final
        resolution['Chemins'].append(trajets_ordonnes[0]['Villes'])

        # Erreur relative de ce chemin
        erreur = 100*(trajets_ordonnes[0]['Distance'] -
                      distance_chemin_optimal)/distance_chemin_optimal
        resolution['Erreur (en %)'].append(erreur)

        # Sélection
        meilleurs_trajets = selection(trajets_ordonnes, POURCENTAGE_SELECTION)

        # Génération
        trajets_initiaux = generation(
            meilleurs_trajets, NOMBRE_TRAJET, POURCENTAGE_MUTATION, matrice_distance)

    resolution['Temps de calcul (en s)'] = time.time() - start
    return resolution
