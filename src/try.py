from distance import distance_trajet, matrice_distance
from graph import affichage, representation_itineraire
from kohonen import kohonen, main
from randomData import init_random_df
from testData import data_TSPLIB, tour_optimal, trajet_en_df

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
