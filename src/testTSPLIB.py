from algo2Opt import main
from distance import matrice_distance
from plusProcheVoisin import plus_proche_voisin
from testData import data_TSPLIB, tour_optimal

ensemble_test = ['ulysses22', 'att48', 'berlin52',
                 'st70', 'kroC100', 'gr202', 'ts225', 'a280']


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
        resultats_test.append(main(mat_distance, cheminInitial))
    return (resultats_test)
