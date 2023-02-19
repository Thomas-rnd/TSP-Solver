import algo2Opt
from distance import matrice_distance
import plusProcheVoisin
import geneticAlgorithm
from testData import data_TSPLIB, tour_optimal
from graph import affichage

# Nom des data de test
ENSEMBLE_TEST = ['ulysses22', 'att48', 'berlin52',
                 'st70', 'kroC100', 'ch150', 'gr202', 'tsp225']


def test_global_2_opt():
    """Lancement des tests de l'algorithme 2-opt

    Returns
    -------
    list
        Elle stocke l'ensemble des dictionnaires renvoyés par l'algorithmes testé
    """
    resultats_test = []

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
        cheminInitial, temps_calcul = plusProcheVoisin.plus_proche_voisin(
            data, mat_distance)

        # Lancement de l'algorithme 2-opt
        res = algo2Opt.main(mat_distance, cheminInitial[-1], chemin_optimal)
        resultats_test.append(res)

        # Affichage console des résultats obtenu sur un jeu de donnée
        affichage(res, data)
    return (resultats_test)


def test_global_plus_proche_voisin():
    """Lancement des tests de l'algorithme 2-opt

    Returns
    -------
    list
        Elle stocke l'ensemble des dictionnaires renvoyés par l'algorithmes testé
    """
    resultats_test = []

    for i in ENSEMBLE_TEST:
        # Initialisation du data frame avec TSPLIB
        data = data_TSPLIB(f'../data/{i}.txt')

        # Initialisation de la matrice des distances relatives
        mat_distance = matrice_distance(data)

        # Initialisation du chemin optimal
        chemin_optimal = tour_optimal(f'../data/{i}_opt_tour.txt')

        # Lancement de l'algorithme plus proche voisin
        res = plusProcheVoisin.main(data, mat_distance, chemin_optimal)
        resultats_test.append(res)

        # Affichage console des résultats obtenu sur un jeu de donnée
        affichage(res, data)
    return (resultats_test)


def test_global_algo_genetique():
    """Lancement des tests de l'algorithme génétique

    Returns
    -------
    list
        Elle stocke l'ensemble des dictionnaires renvoyés par l'algorithmes testé
    """
    resultats_test = []

    for i in ENSEMBLE_TEST:
        # Initialisation du data frame avec TSPLIB
        data = data_TSPLIB(f'../data/{i}.txt')

        # Initialisation de la matrice des distances relatives
        mat_distance = matrice_distance(data)

        # Initialisation du chemin optimal
        chemin_optimal = tour_optimal(f'../data/{i}_opt_tour.txt')

        # Lancement de l'algorithme plus proche voisin
        res = geneticAlgorithm.main(data, mat_distance, chemin_optimal)
        resultats_test.append(res)

        # Affichage console des résultats obtenu sur un jeu de donnée
        affichage(res, data)
    return (resultats_test)
