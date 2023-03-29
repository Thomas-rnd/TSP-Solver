import copy
import random
import time

import numpy as np
import pandas as pd

from src.distance import distance_trajet

# Taille de la population initiale
NOMBRE_TRAJET = 100

# Pourcentage de population conservé à chaque épisode
POURCENTAGE_SELECTION = 10/100

# Pourcentage de mutation
POURCENTAGE_MUTATION = 50/100

# Constante permettant d'arrêter la convergence de l'algorithme
NOMBRE_EPOCH = 100


def init_population(nombre_de_trajet: int, data: pd.DataFrame, matrice_distance: np.ndarray) -> list[dict]:
    """Initialisation de la population initiale

    Construction d'une population initiale de N solutions.
    Une solution étant un trajet passant par toutes les villes une et une seule fois

    Parameters
    ----------
    nombre_de_trajet : int
        taille de la population initiale
    data : DataFrame
        dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    list[dict]
        l'ensemble des N trajets distincts crées avec dans le dictionnaire l'information sur
        le chemin `Villes`et la distance `Distance`
    """
    # Liste stockant la population initiale
    trajets = []
    for i in range(nombre_de_trajet):
        # Initialisation de la forme d'une solution. On instancie un nouveau dictionnaire
        # à chaque fois pour éviter les problématiques du type référence
        trajet = {"Villes": [], 'Distance': 0}
        # Génération d'un ordre de parcours des villes de manière aléatoire
        trajet['Villes'] = random.sample(
            range(0, data.shape[0]), data.shape[0])
        # Le marchand revient sur ses pas donc ajout de la première ville à la fin
        # de la liste
        trajet['Villes'].append(trajet['Villes'][0])
        # Calcul de la distance totale du parcours
        trajet = evaluation(trajet, matrice_distance)
        trajets.append(trajet)
    return trajets


def individus_ordonnes(trajets: list[dict]) -> list[dict]:
    """Tri des trajts par ordre croissant de leur distance

    Parameters
    ----------
    trajets : list[dict]
        listes de toute la population de solutions

    Returns
    -------
    list[dict]
        l'ensemble des N trajets triés
    """
    return sorted(trajets, key=lambda x: x['Distance'])


def selection(trajets: list[dict], pourcentage: float) -> list[dict]:
    """Sélection des n meilleurs

    Parmi la population totale on ne conserve qu'un petit pourcentage
    de la population

    Parameters
    ----------
    trajets : list[dict]
        ordre de parcours des villes et distance du trajet
    pourcentage : int
        le pourcentage à garder de la population initiale

    Returns
    -------
    list[dict]
        les n meilleurs de la population initiale
    """
    # Nombre de trajet après sélection
    nombre_selectionne = int(len(trajets)*pourcentage)
    # Récupération de ces n trajets
    trajets = trajets[:nombre_selectionne]
    return trajets


def probabilite(pourcentage: float) -> bool:
    """Tirage aléatoire simulant une probabilité de succés

    Parameters
    ----------
    pourcentage : float
        le pourcentage de succés

    Returns
    -------
    bool
        résultat
    """
    r = random.randint(0, 100)
    if r <= pourcentage:
        return True
    return False


# Pour les mutations il est important de conserver l'intégrité de nos trajets. Le point initial est confondu
# avec le point final. Le choix du point initial est arbitraire, il n'éxiste pas une première ville meilleur que les autres.
# Prenant en compte ce principe, on n'effectura pas de permutation affectant les extrémités du trajet.
def cadre_mutation(nombre_villes: int) -> list[int]:
    """Définition des villes pouvant permuter en fonction de la remarque précédente.

    Les villes qui peuvent permutter sont comprise entre la première et l'avant avant dernière
    On permutte avec la ville suivante donc l'avant dernière permuterait avec la dernière ce qui
    n'est pas possible

    Parameters
    ----------
    nombre_villes : int
        nombre de ville à traverser

    Returns
    -------
    list[int]
        index des villes qui peuvent muter
    """
    # Initialisation du cadre
    villes_mutables = [1, nombre_villes-3]
    return villes_mutables


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
    # Indice des éléments à permuter
    r = random.sample(range(villes_mutables[0], villes_mutables[1]), 2)
    # Permutation des deux éléments
    enfant['Villes'][r[0]], enfant['Villes'][r[1]
                                             ] = enfant['Villes'][r[1]], enfant['Villes'][r[0]]
    return enfant


def generation(trajets_originels: list[dict], nombre_de_trajet: int, pourcentage_mutation: float, matrice_distance: np.ndarray) -> list[dict]:
    """Génération d'une nouvelle population de N trajets

    Generation de m nouveaux trajets pour compléter la population sélectionnée. Ces nouveaux
    trajets sont des enfants des trajets originels

    Parameters
    ----------
    trajets_originels : list[dict]
        ensemble des trajets sélectionnés
    nombre_de_trajet : int
        taille de la population initiale
    pourcentage_mutation : int
        probabilité qu'un trajet mute
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    list[dict]
        population complète
    """
    while len(trajets_originels) < nombre_de_trajet:
        for trajet in trajets_originels:
            # Génération des altérations : mutation, croisement, ...
            if probabilite(pourcentage_mutation):
                trajet = mutation_aleatoire(trajet)
                trajet = evaluation(trajet, matrice_distance)
                trajets_originels.append(trajet)
    return trajets_originels


def evaluation(trajet: dict, matrice_distance: np.ndarray) -> dict:
    """Fonction d'évaluation de l'algorithme

    Evaluation de la population. Plus un trajet est court plus il est considéré comme bon

    Parameters
    ----------
    trajet : dict
        ordre de parcours des villes avec distance relative
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    dict
        un trajet avec sa distance mise à jour
    """
    maj_trajet = copy.deepcopy(trajet)
    maj_trajet['Distance'] = distance_trajet(
        trajet['Villes'], matrice_distance)
    return maj_trajet


def main(data: pd.DataFrame, matrice_distance: np.ndarray, nom_dataset="") -> pd.DataFrame:
    """Lancement de l'algorithme de recherche 

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes
    nom_dataset : str (optionnel)
        Nom du dataset à traiter

    Returns
    -------
    Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """
    # Initialisation de n individus initiaux (Génèse)
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
    solution = trajets_initiaux[0]['Villes']
    distance = distance_trajet(solution, matrice_distance)
    temps_calcul = time.time() - start

    # Création du dataframe à retourner
    df_resultat_test = pd.DataFrame({
        'Algorithme': "genetique",
        'Nom dataset': nom_dataset,
        'Nombre de villes': len(solution)-1,
        # Dans un tableau pour être sur une seule ligne du dataframe
        'Solution': [solution],
        'Distance': distance,
        'Temps de calcul (en s)': temps_calcul
    })

    return df_resultat_test
