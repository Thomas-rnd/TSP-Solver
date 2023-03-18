from distance import distance_trajet, matrice_distance
from affichage_resultats import affichage, representation_itineraire_back
from kohonen_fail import kohonen, main
from init_random_data import init_random_df
from init_test_data import data_TSPLIB, tour_optimal, trajet_en_df

# Initialisation du data frame avec TSPLIB
# data = data_TSPLIB('../data/tsp225.txt')
data = data_TSPLIB()
# data=init_random_df(100)

# Initialisation de la matrice des distances relatives#
mat_distance = matrice_distance(data)

# Chemin le plus court
# chemin_optimal = tour_optimal('../data/tsp225_opt_tour.txt')
chemin_optimal = tour_optimal()

res = main(data, mat_distance, chemin_optimal)
# affichage(res, data)
