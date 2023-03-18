import copy
import random
import time

import numpy as np
import pandas as pd

from distance import distance_trajet_mat

"""
En s'inspirant des cours dispensés par M. Jean-Marc Salotti en deuxième année
à l'ENSC
"""

# Taille de la population initiale
NOMBRE_TRAJET = 100

# Pourcentage de population conservé à chaque epoch
POURCENTAGE_SELECTION = 10/100

# Pourcentage de mutation
POURCENTAGE_MUTATION = 20/100

# Constante permettant d'arrêter la convergence de l'algorithme
NOMBRE_EPOCH = 200


def init_population(nombre_de_trajet: int, data: pd.DataFrame, matrice_distance: pd.DataFrame) -> list:
    """Initialisation de la population initiale 

    Construction d'une population initiale de N solutions.
    Une solution étant un trajet passant par toutes les villes une et une seule fois

    Parameters
    ----------
    nombre_de_trajet : int
        taille de la population initiale
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : DataFrame
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    list
        L'ensemble des N trajets distincts crées
    """
    # Liste des trajets initiaux
    trajets = []
    for i in range(nombre_de_trajet):
        # Initialisation de la forme d'une solution
        # Dans la boucle pour éviter les problèmes du type référence
        trajet = {"Villes": [], 'Distance': 0}
        # Génération d'un ordre de parcours des villes de manière aléatoire
        trajet['Villes'] = random.sample(
            range(0, data.shape[0]), data.shape[0])
        # Le marchand revient sur ses pas
        trajet['Villes'].append(trajet['Villes'][0])
        # Calcul de la distance total du parcours
        trajet = evaluation(trajet, matrice_distance)
        trajets.append(trajet)
    return trajets


def individus_ordonnes(trajets: list) -> list:
    """Tri les trajets par ordre croissant de leur distance 

    Parameters
    ----------
    trajets : list
        liste des différents trajets

    Returns
    -------
    list
        l'ensemble des N trajets triés
    """
    return sorted(trajets, key=lambda x: x['Distance'])


def selection(trajets: list, pourcentage: int):
    """Sélection des n% meilleurs

    Parmi la population totale on ne conserve qu'un petit pourcentage
    des meilleurs

    Parameters
    ----------
    trajets : list
        liste des différents trajets
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


def probabilite(pourcentage: int) -> bool:
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


# Pour les mutations il est important de conserver l'intégrité de nos trajets. Le point initial est confondu
# avec le point final. Le choix du point initial est arbitraire, il n'éxiste pas une première ville meilleur que les autres.
# Prenant en compte ce principe, on effectura pas de permutation affectant les extrémités du trajet.


def cadre_mutation(nb_villes: int) -> list:
    """Définition des villes pouvant permuter en fonction de la remarque précédente.

    Les villes pouvant permutter sont comprises entre la première et l'avant avant dernière.
    On permutte avec la ville suivante donc l'avant dernière permuterait avec la dernière ce qui 
    n'est pas possible

    Parameters
    ----------
    nb_villes : int
        nombre de ville à traverser 

    Returns
    -------
    list
        index des villes qui peuvent muter
    """
    # Initialisation du cadre
    villes_mutables = [1, nb_villes-3]
    return villes_mutables


def mutation_sucessive(trajet: dict) -> dict:
    """Définition d'une mutation d'un individu

    Cette mutation est une permutation aléatoire de deux villes successives

    Parameters
    ----------
    trajet : dict
        ordre de parcours des villes et distance du trajet

    Returns
    -------
    dict
        nouvel ordre de parcours des villes après mutation
    """
    # Création d'un nouveau dictionnaire pour stocker le trajet muté
    enfant = {'Villes': [], 'Distance': 0}
    villes_mutables = cadre_mutation(len(trajet['Villes']))
    # Je ne veux pas modifier le trajet initial. Comme c'est un type référence
    # je réalise une copie particulière pour ne pas pointer vers la même adresse mémoire
    enfant['Villes'] = copy.deepcopy(trajet['Villes'])
    # Indice de l'élément à permuter avec le suivant
    r = random.randint(villes_mutables[0], villes_mutables[1])
    # Permutation des deux éléments
    enfant['Villes'][r], enfant['Villes'][r +
                                          1] = enfant['Villes'][r+1], enfant['Villes'][r]
    return enfant


def mutation_aleatoire(trajet: dict) -> dict:
    """Définition d'une mutation d'un individu

    Cette mutation est une permutation aléatoire de deux villes

    Parameters
    ----------
    trajet : dict
        ordre de parcours des villes et distance du trajet

    Returns
    -------
    dict
        nouvel ordre de parcours des villes après mutation
    """
    # Création d'un nouveau dictionnaire pour stocker le trajet muté
    enfant = {'Villes': [], 'Distance': 0}
    villes_mutables = cadre_mutation(len(trajet['Villes']))
    # Je ne veux pas modifier le trajet initial. Comme c'est un type référence
    # je réalise une copie particulière pour ne pas pointer vers la même adresse mémoire
    enfant['Villes'] = copy.deepcopy(trajet['Villes'])
    # Indice des éléments à permuter cette fois si ils ne sont pas forcément l'un après l'autre
    r = random.sample(range(villes_mutables[0], villes_mutables[1]), 2)
    # Permutation des deux éléments
    enfant['Villes'][r[0]], enfant['Villes'][r[1]
                                             ] = enfant['Villes'][r[1]], enfant['Villes'][r[0]]
    return enfant


def generation(trajets_originels: list, nombre_de_trajet: int, pourcentage_mutation: int, matrice_distance: pd.DataFrame) -> list:
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
    # Tant que la population n'a pas la bonne taille
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


def evaluation(trajet: list, matrice_distance: pd.DataFrame) -> dict:
    """Fonction d'évaluation de l'algorithme

    On va venir noter chacun des trajets de la population. Plus un trajet
    est court plus il est considéré comme bon

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
    maj_trajet['Distance'] = distance_trajet_mat(
        trajet['Villes'], matrice_distance)
    return maj_trajet


def main(data: pd.DataFrame, matrice_distance: pd.DataFrame) -> pd.DataFrame:
    """Lancement de l'algorithme de recherche sur 1 jeu de données

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
    # Initialisation de n individus initiaux (génèse)
    trajets_initiaux = init_population(NOMBRE_TRAJET, data, matrice_distance)

    # Initialisation du nombre d'epoch
    epoch = 0

    # Evaluation du temps de calcul
    start = time.time()
    # On arrete l'algorithme après un nombre d'epoch fixé. Pour permettre de visualiser
    # un résultat même si l'algorithme ne converge pas (ou pas assez vite).
    while epoch <= NOMBRE_EPOCH:
        epoch += 1
        # Tri
        trajets_ordonnes = individus_ordonnes(trajets_initiaux)

        # Sélection
        meilleurs_trajets = selection(trajets_ordonnes, POURCENTAGE_SELECTION)

        # Génération
        trajets_initiaux = generation(
            meilleurs_trajets, NOMBRE_TRAJET, POURCENTAGE_MUTATION, matrice_distance)

    # Chemin final trouvé
    solution = meilleurs_trajets[0]['Villes']
    distance = meilleurs_trajets[0]['Distance']
    temps_calcul = time.time() - start

    # Création du dataframe à retourner
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': len(solution),
        # Dans un tableau pour être sur une seule ligne du dataframe
        'Solution': [solution],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Distance': distance,
        'Temps de calcul (en s)': temps_calcul
    })

    return df_resultat_test
