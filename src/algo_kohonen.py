import time

import numpy as np
import pandas as pd

from src.affichage_resultats import representation_itineraire_web, representation_reseau
from src.distance import distance_euclidienne, distance_trajet, neurone_gagnant
from src.init_test_data import data_TSPLIB, normalisation


def creation_reseau(taille: int) -> np.array:
    """
    Création d'un réseau d'un taille donnée

    Parameters
    ----------
    taille : int 
        nombre de neuronnes à créer

    Returns
    -------
    np.array
        un ensemble de taille neurones de dimension 2 dans l'intervalle [0,1)
    """
    return np.random.rand(taille, 2)


def voisinage(index_neuronne_gagnant: int, rayon: int, nombre_neurones: int) -> np.array:
    """Génération d'une gaussienne à valeur dans [0,1] centrée en index_neuronne_gagnant
    et d'écart type rayon. Cette gaussienne permet de modéliser l'attirance du cycle de neuronne.
    Cette fonction est periodique et de période le nombre_neurones.

    Parameters
    ----------
    index_neuronne_gagnant : int 
        index du neuronne gagnant dans le réseaux de kohonen. Moyenne de la gaussienne
    rayon : int 
        écart type de la gaussienne. (rayon d'influence du neuronne gagnant)
    nombre_neurones : int
        nombre de neurones dans le réseau

    Returns
    -------
    np.array
        la gaussienne discrête modélisant l'attraction ainsi crée
    """

    # Impose an upper bound on the radix to prevent NaN and blocks
    if rayon < 1:
        rayon = 1

    # Distance entre les neurones de la chaine et le neurone gagnant
    deltas = np.absolute(index_neuronne_gagnant - np.arange(nombre_neurones))
    distances = np.minimum(deltas, nombre_neurones - deltas)

    # Génération de la distribution gaussienne autour du neurone gagnant
    return np.exp(-(distances*distances) / (2*(rayon*rayon)))


def chemin_final(villes: pd.DataFrame, neurones: np.array) -> list:
    """Recherche du chemin final trouvé par le réseau. 

    Pour cela on atitre à chacune des villes son neurone gagnant et ensuite
    on vient trier les villes dans le même ordre que l'ordre des neurones
    Return the route computed by a network.

    Parameters
    ----------
    villes : DataFrame 
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    neurones : np.array 
        un ensemble de neuronnes de dimension 2 dans l'intervalle [0,1)

    Returns
    -------
    Dataframe
        dataframe final des villes ordonnées
    """
    villes['ordre'] = villes[['x', 'y']].apply(
        lambda c: neurone_gagnant(neurones, c),
        # 1 or 'columns': on applique la fonction à chaque ligne.
        axis=1, raw=True)
    # Retourne un array représentant les données de l'index
    route = villes.sort_values('ordre').index.values
    # On fait attention à fermer le cycle
    return list(np.append(route, route[0]))


def som(data: pd.DataFrame, iterations: int, taux_apprentissage=0.8):
    """Résolution du TSP en utilisant une Self-Organizing Map

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    iterations : int 
        nombre d'itérations maximal
    taux_apprentissage : float
        taux d'apprentissage du réseau de kohonen

    Returns
    -------
    np.array
        la gaussienne discrête modélisant l'attraction ainsi crée

    """
    start_time = time.time()

    # On crée des villes artificielles normalisées
    villes = data.copy()
    villes[['x', 'y']] = normalisation(villes[['x', 'y']])

    # La taille de la population de neuronne est 8 fois celle du nombre de villes
    # Hyperparamètre
    n = villes.shape[0]*8
    # Génération du réseau de neurones
    neurones = creation_reseau(n)
    # print('Réseau de {} neurones créé. On commence les itérations :'.format(n))

    for i in range(iterations):
        if not i % 100:
            # Affichage d'un feeback de l'avancement (avec retour à la ligne)
            # print('\t> Iteration {}/{}'.format(i, iterations), end="\r")

            # Représentation de l'état du réseau
            # representation_reseau(villes, neurones)
            continue

        # On choisit une ville aléatoire. On retourne les valeurs de x et y
        ville = villes.sample(1)[['x', 'y']].values
        index_gagnant = neurone_gagnant(neurones, ville)
        # Génération d'un filtre gaussien modélisant l'attraction entre un neurone et ses voisins
        gaussian = voisinage(index_gagnant, n//10, neurones.shape[0])
        # Mise à jour des poids des neurones (proche de la ville initiale)
        # np.newaxis pour contrôler le broadcasting
        neurones += gaussian[:, np.newaxis] * \
            taux_apprentissage * (ville - neurones)
        # Mise à jour du taux d'apprentissage
        taux_apprentissage = taux_apprentissage * 0.99997
        # Réduction de la distance d'influence d'un neurone
        n = n * 0.9997

        # Si un des paramètres a trop diminué
        if n < 1:
            # print("Le nombre de neurones est trop faible, exécution terminée",
            #      "à l'itération {}".format(i))
            break
        if taux_apprentissage < 0.001:
            # print("Taux d'apprentissage trop faible, exécution terminée",
            #      "à l'itération {}".format(i))
            break

    itineraire = chemin_final(villes, neurones)
    villes = villes.reindex(itineraire)
    temps_calcul = time.time() - start_time

    return itineraire, temps_calcul


def main(data, mat_distance) -> pd.DataFrame:
    """Lancement de l'algorithme de kohonen

    Returns
    -------
    Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """
    # On récupère lechemin trouvé et le temps de résolution de l'algorithme
    itineraire, temps_calcul = som(data, 100000)

    # Dataframe final trouvé. On donne comme nouvel index à data la liste itinéraire
    # solution = data.reindex(itineraire)

    # Calcul de la distance du trajet final trouvé par l'algorithme
    distance_chemin_sub_optimal = distance_trajet(itineraire, mat_distance)

    # Création du dataframe à retourner
    df_resultat_test = pd.DataFrame({
        'Algorithme': "Kohonen",
        'Nombre de villes': len(itineraire),
        # Dans un tableau pour être sur une seule ligne du dataframe
        'Solution': [itineraire],
        # Distance du trajet final
        'Distance': distance_chemin_sub_optimal,
        'Temps de calcul (en s)': temps_calcul
    })

    return df_resultat_test
