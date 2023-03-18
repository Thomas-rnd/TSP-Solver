import algo_2_opt
from distance import matrice_distance
import algo_proche_voisin
import algo_genetique
from init_test_data import data_TSPLIB, tour_optimal
from graph import affichage, representation_temps_calcul, representation_itineraire_web
import pandas as pd

# Nom des data de test
ENSEMBLE_TEST = ['ulysses22', 'att48', 'berlin52',
                 'st70', 'kroC100', 'ch150', 'gr202', 'tsp225']


def test_global_2_opt():
    """Lancement des tests de l'algorithme 2-opt

    Returns
    -------
    Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """
    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': [],
        'Solution': [],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Erreur (en %)': [],
        'Temps de calcul (en s)': []
    })

    for i in ENSEMBLE_TEST:
        # Initialisation du data frame avec TSPLIB
        data = data_TSPLIB(f'../data/{i}.txt')

        # Initialisation de la matrice des distances relatives
        mat_distance = matrice_distance(data)

        # Initialisation du chemin optimal
        chemin_optimal = tour_optimal(f'../data/{i}_opt_tour.txt')

        # On prend un chemin initial meilleur qu'un chemin aléatoire
        # Attention cheminInitial est la liste des chemin exploré par l'algorithme
        # plus_proche_voisin
        cheminInitial, temps_calcul = algo_proche_voisin.plus_proche_voisin(
            data, mat_distance)

        # Lancement de l'algorithme 2-opt
        df_res = algo_2_opt.main(
            mat_distance, cheminInitial[-1], chemin_optimal)

        # Affichage des résultats obtenu sur un jeu de donnée
        affichage(df_res, data)
        df_resultat_test = pd.concat(
            [df_resultat_test, df_res], ignore_index=True)
    representation_temps_calcul(df_resultat_test)
    return (df_resultat_test)


def test_unitaire_2_opt(num_dataset):
    """Lancement d'un test de l'algorithme 2-opt

    Returns
    -------
    Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """
    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': [],
        'Solution': [],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Erreur (en %)': [],
        'Temps de calcul (en s)': []
    })

    # Initialisation du data frame avec TSPLIB
    data = data_TSPLIB(f'../data/{ENSEMBLE_TEST[num_dataset]}.txt')

    # Initialisation de la matrice des distances relatives
    mat_distance = matrice_distance(data)

    # Initialisation du chemin optimal
    chemin_optimal = tour_optimal(
        f'../data/{ENSEMBLE_TEST[num_dataset]}_opt_tour.txt')

    # On prend un chemin initial meilleur qu'un chemin aléatoire
    # Attention cheminInitial est la liste des chemin exploré par l'algorithme
    # plus_proche_voisin
    cheminInitial, temps_calcul = algo_proche_voisin.plus_proche_voisin(
        data, mat_distance)

    # Lancement de l'algorithme 2-opt
    df_res = algo_2_opt.main(mat_distance, cheminInitial[-1], chemin_optimal)

    df_resultat_test = pd.concat(
        [df_resultat_test, df_res], ignore_index=True)
    print(df_resultat_test)
    return (df_resultat_test, data)


def test_global_plus_proche_voisin():
    """Lancement des tests de l'algorithme plus proche voisin

    Returns
    -------
    Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """
    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': [],
        'Solution': [],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Erreur (en %)': [],
        'Temps de calcul (en s)': []
    })

    for i in ENSEMBLE_TEST:
        # Initialisation du data frame avec TSPLIB
        data = data_TSPLIB(f'../data/{i}.txt')

        # Initialisation de la matrice des distances relatives
        mat_distance = matrice_distance(data)

        # Initialisation du chemin optimal
        chemin_optimal = tour_optimal(f'../data/{i}_opt_tour.txt')

        # Lancement de l'algorithme plus proche voisin
        df_res = algo_proche_voisin.main(data, mat_distance, chemin_optimal)

        # Affichage des résultats obtenu sur un jeu de donnée
        affichage(df_res, data)
        df_resultat_test = pd.concat(
            [df_resultat_test, df_res], ignore_index=True)
    representation_temps_calcul(df_resultat_test)
    return (df_resultat_test)


def test_global_algo_genetique():
    """Lancement des tests de l'algorithme plus proche voisin

    Returns
    -------
    Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """
    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        'Nombre de villes': [],
        'Solution': [],
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Erreur (en %)': [],
        'Temps de calcul (en s)': []
    })

    for i in ENSEMBLE_TEST:
        # Initialisation du data frame avec TSPLIB
        data = data_TSPLIB(f'../data/{i}.txt')

        # Initialisation de la matrice des distances relatives
        mat_distance = matrice_distance(data)

        # Initialisation du chemin optimal
        chemin_optimal = tour_optimal(f'../data/{i}_opt_tour.txt')

        # Lancement de l'algorithme plus proche voisin
        df_res = algo_genetique.main(data, mat_distance, chemin_optimal)

        # Affichage des résultats obtenu sur un jeu de donnée
        affichage(df_res, data)
        df_resultat_test = pd.concat(
            [df_resultat_test, df_res], ignore_index=True)
    representation_temps_calcul(df_resultat_test)
    return (df_resultat_test)
