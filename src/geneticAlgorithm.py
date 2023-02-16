import copy
import random
import time

import matplotlib.pyplot as plt
import numpy
import pandas as pd
from distance import distance_trajet, matrice_distance
from graph import representation_itineraire
from testData import data_TSPLIB, tour_optimal, trajet_en_df

# Taille de la population
NOMBRE_TRAJET = 500
# Pourcentage de population conservé à chaque épisode
POURCENTAGE_SELECTION = 10/100

# Pourcentage de mutation
POURCENTAGE_MUTATION = 20/100


def init_population(nombre_de_trajet, data, matrice_distance):
    """Construction d'une population initiale de N solutions.
    Une solution étant un trajet passant par toutes les villes une et une seule fois"""
    # Liste des solutions
    trajets = []
    # Initialisation de la forme d'une solution
    for i in range(nombre_de_trajet):
        # On instancie un nouveau dictionnaire car type référence
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
    """Rangement des trajets"""
    return sorted(trajets, key=lambda x: x['Distance'])


def selection(trajets, pourcentage):
    """On ne conserve que les n meillleurs"""
    nombre_selectionne = int(len(trajets)*pourcentage)
    trajets = trajets[:nombre_selectionne]
    return trajets


def probabilite(pourcentage):
    r = random.randint(0, 100)
    if r <= pourcentage:
        return True
    return False


"""
Pour les mutations il est important de conserver l'intégrité de nos trajets. Le point initial est confondu
avec le point final. Le choix du point initial est arbitraire, il n'éxiste pas un premier meilleurs que les autres,
en effet on doit forcément passer par toute les villes. Prenant en compte ce principe, on effectura pas de permutation
affectant les extrémités du trajet.
"""


def cadre_mutation(nbVilles):
    """Les villes qui peuvent permutter sont comprise entre la première et l'avant avant dernière
    On permutte avec la ville suivante donc l'avant dernière permuterait avec la dernière pas possible"""
    villesMutables = [1, nbVilles-3]
    return villesMutables


def mutation_sucessive(trajet):
    """Permutation aléatoire de deux villes successives"""
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
    """Permutation aléatoire de deux villes"""
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


def generation(data, trajets_originels, nombre_de_trajet, pourcentage_mutation, matrice_distance):
    """Generation de m nouveaux trajets pour compléter la population. Ces nouveaux
    trajets sont sont des enfants des trajets sélectionnés"""
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
    """Evaluation de la population. Plus un trajet est court plus il est considéré comme bon"""
    maj_trajet = copy.deepcopy(trajet)
    maj_trajet['Distance'] = distance_trajet(
        trajet['Villes'], matrice_distance)
    return maj_trajet


def affichage(resolution, data):
    # Affichage pour observer ou non la convergence
    df_meilleur_trajet = trajet_en_df(resolution['Chemins'][-1], data)
    representation_itineraire(df_meilleur_trajet)


def main(data, matrice_distance, chemin):
    start = time.time()

    distance_chemin_optimal = distance_trajet(chemin, matrice_distance)

    # Initialisation de n individus initiaux (Génèse)
    trajets_initiaux = init_population(NOMBRE_TRAJET, data, matrice_distance)
    resolution = {
        'Nombre de villes': len(trajets_initiaux),
        'Algorithme': 'Algorithme génétique',
        'Distance': 'Euclidienne-2D',
        'Chemins': [],
        'Chemin optimal': chemin,
        'Erreur (en %)': [100],
        'Temps de calcul': 0
    }
    # On arrète l'algorithme quand une approximation est atteinte
    while resolution['Erreur (en %)'][-1] > 3:
        # Tri
        trajets_ordonnes = individus_ordonnes(trajets_initiaux)

        resolution['Chemins'].append(trajets_ordonnes[0]['Villes'])
        erreur = 100*(trajets_ordonnes[0]['Distance'] -
                      distance_chemin_optimal)/distance_chemin_optimal
        resolution['Erreur (en %)'].append(erreur)

        # Sélection
        meilleurs_trajets = selection(trajets_ordonnes, POURCENTAGE_SELECTION)
        # Generation
        trajets_initiaux = generation(
            data, meilleurs_trajets, NOMBRE_TRAJET, POURCENTAGE_MUTATION, matrice_distance)

    resolution['Temps de calcul'] = time.ctime(time.time() - start)
    return resolution
