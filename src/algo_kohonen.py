import time

import numpy as np
import pandas as pd

from src.affichage_resultats import representation_reseau
from src.distance import distance_trajet, neurone_gagnant
from src.init_test_data import normalisation

# En s'inspirant des cours dispensés à l'ENSC en apprentissage automatique j'ai essayé
# de mettre en place une carte auto-génératrice afin de résoudre le TSP.

# Afin d'implémenter cette algorithme complexe je suis inspiré de recherches sur le sujet
# Cf. https://github.com/diego-vicente/som-tsp
# Cf. https://github.com/sdpython/ensae_teaching_cs/blob/be65e97cf24abf05cb3471f3989cb7c7d5938236/src/ensae_teaching_cs/special/tsp_kohonen.py#L202


def creation_reseau(taille: int) -> np.ndarray:
    """
    Création d'un réseau d'un taille donnée. Le réseau est une suite 1D de neurones

    Parameters
    ----------
    taille : int 
        nombre de neuronnes à créer

    Returns
    -------
    np.ndarray
        un vecteur de dimension `taille` composé de neurones à 2 dimension à valeur dans l'intervalle [0,1)
    """
    return np.random.rand(taille, 2)


def voisinage(index_neuronne_gagnant: int, rayon: float, nombre_neurones: int) -> np.ndarray:
    """Génération d'une gaussienne à valeur dans [0,1] centrée en `index_neuronne_gagnant`
    et d'écart type `rayon`. Cette gaussienne permet de modéliser l'attirance du cycle de neuronne.

    Cette fonction est periodique et de période le nombre_neurones (attirance dans un cycle).

    Parameters
    ----------
    index_neuronne_gagnant : int 
        index du neuronne gagnant dans le réseaux de kohonen. Moyenne de la gaussienne
    rayon : float 
        écart type de la gaussienne (rayon d'influence du neuronne gagnant)
    nombre_neurones : int
        nombre de neurones dans le réseau

    Returns
    -------
    np.ndarray
        la gaussienne discrête modélisant l'attraction dans le réseau de neurones
    """

    # L'écart type ne peut pas être inférieur 1 pour prévenir de valeurs NaN
    if rayon < 1:
        rayon = 1

    # Distance entre les neurones de la chaine et le neurone gagnant (en 1D on utilise la norme 1)
    deltas = np.absolute(index_neuronne_gagnant - np.arange(nombre_neurones))
    # Ajout de la prise en compte du cycle dans le calcul des distances
    distances = np.minimum(deltas, nombre_neurones - deltas)

    # Génération de la distribution gaussienne autour du neurone gagnant
    return np.exp(-(distances*distances) / (2*(rayon*rayon)))  # type: ignore


def chemin_final(villes: pd.DataFrame, neurones: np.ndarray) -> list[int]:
    """Recherche du chemin final trouvé par le réseau. 

    Pour cela on attribut à chacune des villes son neurone gagnant et ensuite
    on vient trier les villes dans le même ordre que celui effectif dans le réseau

    Parameters
    ----------
    villes : DataFrame 
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    neurones : np.ndarray 
        un ensemble de neuronnes de dimension 2 dans l'intervalle [0,1)

    Returns
    -------
    Dataframe
        dataframe final des villes ordonnées
    """
    villes['ordre'] = villes[['x', 'y']].apply(
        # convertion en int et non plus intp
        lambda c: int(neurone_gagnant(neurones, c)),
        # 1 or 'columns': on applique la fonction à chaque ligne.
        axis=1, raw=True)
    # Retourne un array représentant les données de l'index
    route = villes.sort_values('ordre').index.values
    # On fait attention à fermer le cycle
    return list(np.append(route, route[0]))


def carte_auto_adaptatives(data: pd.DataFrame, iterations: int, taux_apprentissage=0.8) -> tuple[list[int], float, list[np.ndarray]]:
    """Résolution du TSP en utilisant une Cartes auto-adaptatives

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
    itineraire : list[int]
        le chemin final trouvé
    temps_calcul : float
        temps necessaire à la résolution du problème
    evolution_reseau : list[np.ndarray]
        stockage de l'évolution du réseau de neurones
    """
    start_time = time.time()

    # On crée des villes artificielles normalisées
    villes = data.copy()
    villes[['x', 'y']] = normalisation(villes[['x', 'y']])

    # Hyperparamètre
    # La taille de la population de neuronne est 8 fois celle du nombre de villes
    n = villes.shape[0]*8
    # Génération du réseau de neurones
    neurones = creation_reseau(n)
    # print('Réseau de {} neurones créé. On commence les itérations :'.format(n))

    # Stockage de l'évolution du réseau de neurones
    evolution_reseau = []

    for i in range(1, iterations):
        # Intervalle de sauvegarde du réseau
        if not i % 1000:
            # Représentation de l'état du réseau
            # representation_reseau(villes, neurones).show()
            evolution_reseau.append(neurones.copy())

        # On choisit une ville aléatoire. On retourne les valeurs de x et y
        ville = villes.sample(1)[['x', 'y']].values
        index_gagnant = neurone_gagnant(neurones, ville)
        # Génération d'un filtre gaussien modélisant l'attraction entre un neurone et ses voisins
        gaussian = voisinage(int(index_gagnant), n//10, neurones.shape[0])
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
    temps_calcul = time.time() - start_time

    return itineraire, temps_calcul, evolution_reseau


def main(data: pd.DataFrame, mat_distance: np.ndarray, nom_dataset="") -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Lancement de l'algorithme de kohonen

    Parameters
    ----------
    data : DataFrame
        dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes
    nom_dataset : str (optionnel)
        nom du dataset à traiter

    Returns
    -------
    df_resultat_test : Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    evolution_reseau : list
        variable retraçant l'évolution du réseau de neurones
    """
    # Résolution du TSP
    itineraire, temps_calcul, evolution_reseau = carte_auto_adaptatives(
        data, 100000)

    # Calcul de la distance du trajet final trouvé par l'algorithme
    distance_chemin_sub_optimal = distance_trajet(itineraire, mat_distance)

    # Création du dataframe à retourner
    # On inclut pas l'évolution du réseau pour pas sucharger le fichier csv de résultats
    df_resultat_test = pd.DataFrame({
        'Algorithme': "kohonen",
        'Nom dataset': nom_dataset,
        'Nombre de villes': len(itineraire)-1,
        # Dans un tableau pour être sur une seule ligne du dataframe
        'Solution': [itineraire],
        'Distance': distance_chemin_sub_optimal,
        'Temps de calcul (en s)': temps_calcul
    })

    return df_resultat_test, evolution_reseau
