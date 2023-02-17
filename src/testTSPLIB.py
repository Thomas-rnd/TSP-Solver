from algo2Opt import main
from distance import matrice_distance
from plusProcheVoisin import plus_proche_voisin
from testData import data_TSPLIB, tour_optimal
from graph import affichage

ensemble_test = ['ulysses22', 'att48', 'berlin52',
                 'st70', 'kroC100', 'ch150', 'gr202', 'tsp225']


def test_global():
    resultats_test = []

    for i in ensemble_test:
        # Initialisation du data frame avec TSPLIB
        data = data_TSPLIB(f'../data/{i}.txt')

        # Initialisation de la matrice des distances relatives
        mat_distance = matrice_distance(data)

        # Chemin le plus court
        chemin = tour_optimal(f'../data/{i}_opt_tour.txt')

        # On prend un chemin initial meilleur qu'un chemin aléatoire
        cheminInitial = plus_proche_voisin(data, mat_distance)

        # Lancement de l'algorithme 2-opt
        res = main(mat_distance, cheminInitial, chemin)
        resultats_test.append(res)
        affichage(res, data)
    return (resultats_test)
