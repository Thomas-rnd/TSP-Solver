import pandas as pd

import algo_2_opt
import algo_genetique
import algo_proche_voisin
from distance import matrice_distance
from init_test_data import data_TSPLIB

# Nom des data de test
ENSEMBLE_TEST = ['dj38', 'qa194', 'ar9152']


def test_global_2_opt():
    """Lancement des tests de l'algorithme 2-opt

    Returns
    -------
    Dataframe 
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme. Un ligne représente un test sur un jeu de données
    """
    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': [],
        'Solution': [],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Erreur (en %)': [],
        'Temps de calcul (en s)': []
    })
    # Test sur l'ensemble des data
    for num_dataset in range(len(ENSEMBLE_TEST)):
        df_res, data = test_unitaire_2_opt(num_dataset)
        df_resultat_test = pd.concat(
            [df_resultat_test, df_res], ignore_index=True)

    return df_resultat_test


def test_unitaire_2_opt(num_dataset):
    """Lancement d'un test de l'algorithme 2-opt

    Parameters
    ----------
    num_dataset : int
        Numéro du dataset sur lequel est réalisé le test. Ce numéro est égale à son 
        index dans ENSEMBLE_TEST

    Returns
    -------
    Dataframe
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme
    """
    # Initialisation du dataframe avec TSPLIB
    data = data_TSPLIB(f'../data/{ENSEMBLE_TEST[num_dataset]}.tsp')

    # Initialisation de la matrice des distances relatives
    mat_distance = matrice_distance(data)

    # On prend un chemin initial meilleur qu'un chemin aléatoire
    # Attention chemin_initial est la liste des chemins explorés par l'algorithme
    # plus_proche_voisin
    chemin_initial, temps_calcul = algo_proche_voisin.plus_proche_voisin(
        mat_distance)

    # Lancement de l'algorithme 2-opt
    df_res = algo_2_opt.main(
        data, mat_distance, chemin_initial)
    return df_res


def test_global_plus_proche_voisin():
    """Lancement des tests de l'algorithme plus proche voisin

    Returns
    -------
    Dataframe 
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme. Un ligne représente un test sur un jeu de données
    """
    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': [],
        'Solution': [],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Distance': [],
        'Temps de calcul (en s)': []
    })

    for num_dataset in range(len(ENSEMBLE_TEST)):
        df_res = test_unitaire_plus_proche_voisin(num_dataset)
        df_resultat_test = pd.concat(
            [df_resultat_test, df_res], ignore_index=True)

    return df_resultat_test


def test_unitaire_plus_proche_voisin(num_dataset: int) -> pd.DataFrame:
    """Lancement d'un test de l'algorithme du plus proche voisin

    Parameters
    ----------
    num_dataset : int
        Numéro de dataset sur lequel est réalisé le test. Ce numéro est égale à son 
        index dans ENSEMBLE_TEST

    Returns
    -------
    Dataframe
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme
    """
    # Initialisation du data frame avec TSPLIB
    data = data_TSPLIB(f'../data/{ENSEMBLE_TEST[num_dataset]}.tsp')

    # Initialisation de la matrice des distances relatives
    mat_distance = matrice_distance(data)

    # Lancement de l'algorithme plus proche voisin
    df_res = algo_proche_voisin.main(data, mat_distance)

    return (df_res)


def test_global_algo_genetique():
    """Lancement des tests de l'algorithme génétique

    Returns
    -------
    Dataframe 
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme. Un ligne représente un test sur un jeu de données
    """
    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': [],
        'Solution': [],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Erreur (en %)': [],
        'Temps de calcul (en s)': []
    })

    for num_dataset in range(len(ENSEMBLE_TEST)):
        df_res, data = test_unitaire_algo_genetique(num_dataset)
        df_resultat_test = pd.concat(
            [df_resultat_test, df_res], ignore_index=True)

    return df_resultat_test


def test_unitaire_algo_genetique(num_dataset):
    """Lancement d'un test de l'algorithme génétique

    Parameters
    ----------
    num_dataset : int
        Numéro de dataset sur lequel est réalisé le test. Ce numéro est égale à son 
        index dans ENSEMBLE_TEST

    Returns
    -------
    Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """
    # Initialisation du data frame avec TSPLIB
    data = data_TSPLIB(f'data/{ENSEMBLE_TEST[num_dataset]}.txt')

    # Initialisation de la matrice des distances relatives
    mat_distance = matrice_distance(data)

    # Initialisation du chemin optimal
    chemin_optimal = tour_optimal(
        f'data/{ENSEMBLE_TEST[num_dataset]}_opt_tour.txt')

    # Lancement de l'algorithme génétique
    df_res = src.algo_genetique.main(data, mat_distance, chemin_optimal)

    return (df_res, data)
