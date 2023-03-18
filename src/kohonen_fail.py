import random
import time

from distance import distance_euclidienne, distance_trajet
from affichage_resultats import affichage, representation_itineraire_back

# Nombre de neurones initialement dans la carte
N = 4

# Coefficients d'apprentissage
E0 = 99
D0 = 90
N0 = 10


def barycentre(data):
    """
    Les coordonnées X et Y du barycentre s'obtiennent en sommant les coordonnées
    pondérées de chaque site et en les divisant par la somme des pondérations.

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir

    Returns
    -------
    x : bouble
        cordonnée X du barycentre
    y : double
        cordonnée Y du barycentre
    """
    x = sum(data.iloc[0, :])/len(data.iloc[0, :])
    y = sum(data.iloc[1, :])/len(data.iloc[1, :])
    return x, y


def initialisation_neurones(x, y, n=3):
    """
    n (>=3) neurones sont initialisés avec des poids (w_i,w_j) proches du barycentre.
    Chaque neurones à maximum 2 voisins (un à droite et un à gauche) et minimum 1
    s'il est sur une extrémité.

    Parameters
    ----------
    x : bouble
        cordonnée X du barycentre
    y : double
        cordonnée Y du barycentre
    n : int
        nombre de neurones initial

    Returns
    -------
    list
        liste de neurones initialisés
    """
    neurones = []
    while len(neurones) < n:
        zone_autour_barycentre = 0.5

        # Création aléatoire de poids dans une zone de plus ou moins
        # 10% autour du barycentreune fois projeté dans l'espace de
        # des données.
        w_i = x+zone_autour_barycentre*random.uniform(-x, x)
        w_j = y+zone_autour_barycentre*random.uniform(-y, y)
        neurone = [w_i, w_j]
        if neurone not in neurones:
            neurones.append(neurone)
    return neurones


def le_plus_proche(neurones, ville):
    """
    Recherche le neurone le plus proche de la ville passée en entrée

    Parameters
    ----------
    neurones : list
        l'ensemble des neurones du réseau de kohonen
    ville : list
        contient les coordonnées x,y de la ville à étudier

    Returns
    -------
    index : int
        index du neurone gagnant
    poids : list
        poids du neurone gagnant
    """
    # Initialisation en fonction du premier neurone
    index = 0
    poids = neurones[0]
    dist = distance_euclidienne(
        ville[0], ville[1], neurones[0][0], neurones[0][1])

    for i in range(1, len(neurones)):
        d = distance_euclidienne(
            ville[0], ville[1], neurones[i][0], neurones[i][1])
        if d < dist:
            dist = d
            index = i
            poids = neurones[i]
    return index, poids


def maj_poids(neurone, ville, epoch):
    """
    Mise à jour des poids du neurone gagnant à une certaine epoch de la
    phase d'entraînement

    Parameters
    ----------
    neurone : list
        poids du neuronne gagnant
    ville : list
        contient les coordonnées x,y de la ville à étudier
    epoch : int
        période dans la phase d'entraînement

    Returns
    -------
    list
        poids mise à jour du neurone gagnant
    """
    distance = distance_euclidienne(ville[0], ville[1], neurone[0], neurone[1])

    coef_apprentissage = apprentissage_direct(epoch, distance)
    erreur_position = soustraction_vecteurs(ville, neurone)
    ajustement_position = multiplier_un_vecteur(
        erreur_position, coef_apprentissage)
    nouveau_poids = addition_vecteurs(neurone, ajustement_position)

    return nouveau_poids


def maj_poids_voisin(neurone_gagnant, voisin, ville, epoch):
    """
    Mise à jour des poids des neurones voisins au neurone gagnant à une
    certaine epoch de la phase d'entraînement

    Parameters
    ----------
    neurone_gagnant : list
        poids du neuronne
    voisin : list
        poids du neuronne
    ville : list
        contient les coordonnées x,y de la ville à étudier
    epoch : int
        période dans la phase d'entraînement

    Returns
    -------
    list
        poids mise à jour du voisin
    """
    distance_neurone_ville = distance_euclidienne(
        voisin[0], voisin[1], neurone_gagnant[0], neurone_gagnant[1])
    distance_neurone_voisin = distance_euclidienne(
        neurone_gagnant[0], neurone_gagnant[1], voisin[0], voisin[1])

    coef_apprentissage = apprentissage_voisin(
        epoch, distance_neurone_ville, distance_neurone_voisin)
    erreur_position = soustraction_vecteurs(
        ville, neurone_gagnant)
    ajustement_position = multiplier_un_vecteur(
        erreur_position, coef_apprentissage)
    nouveau_poids = addition_vecteurs(voisin, ajustement_position)

    return nouveau_poids


def ajout_neurone(neurone_dupliqué):
    """
    Ajout d'un neurone dans le chemin, il est placé à proximité du neurone dupliqué

    Parameters
    ----------
    neurone_dupliqué : list
        poids du neuronne

    Returns
    -------
    list
        poids du neurone ajouté
    """
    zone_autour_dupliqué = 0.01

    # Création aléatoire de poids dans une zone de plus ou moins
    # 10% autour du barycentreune fois projeté dans l'espace de
    # des données.
    x = neurone_dupliqué[0]
    y = neurone_dupliqué[1]
    w_i = x+zone_autour_dupliqué*random.uniform(-x, x)
    w_j = y+zone_autour_dupliqué*random.uniform(-y, y)
    return [w_i, w_j]


def apprentissage_direct(epoch, distance):
    """
    Définition du coefficient d'apprentissage

    Parameters
    ----------
    epoch : int
        période dans la phase d'entraînement
    distance : double
        distance séparant la ville du neurone gagnant

    Returns
    -------
    double
        coefficient d'apprentissage
    """

    return ((E0/(1+epoch))*(D0/(D0+distance)))


def apprentissage_voisin(epoch, distance_neurone_ville, distance_neurone_voisin):
    """
    Définition du coefficient d'apprentissage

    Parameters
    ----------
    epoch : int
        période dans la phase d'entraînement
    distance_neurone_ville : double
        distance séparant la ville du neurone gagnant
    distance_neurone_voisin : double
        distance séparant le neurone gagnant d'un de ses voisins

    Returns
    -------
    double
        coefficient d'apprentissage
    """
    return ((N0/(1+epoch))*apprentissage_direct(epoch, distance_neurone_ville))


def soustraction_vecteurs(v, n):
    """
    Soustraction de deux vecteurs.

    Parameters
    ----------
    v : list
        vecteur 1
    n : list
        vecteur 2

    Returns
    -------
    list
        vecteur issu de la soustraction
    """
    return [v[0] - n[0], v[1] - n[1]]


def addition_vecteurs(v, n):
    """
    Addition de deux vecteurs.

    Parameters
    ----------
    v : list
        vecteur 1
    n : list
        vecteur 2

    Returns
    -------
    list
        vecteur issu de l'addition
    """
    return [v[0] + n[0], v[1] + n[1]]


def multiplier_un_vecteur(v, f):
    """
    Multiplication d'un vecteur par une constante.

    Parameters
    ----------
    v : list
        vecteur
    f : double
        constante de multiplication

    Returns
    -------
    list
        vecteur issu de la multiplication
    """
    return [v[0] * f, v[1] * f]


def kohonen(data, neurones):
    """
    Implémentation de l'algorithme de kohonen

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    neurones : list
        réseau de neurones de kohonen

    Returns
    -------
    list
        Ensemble des états du réseau de neurones jusqu'à convergence finale
    """
    epoch = 0
    entrainement = True
    # Stockage de l'avancement de la convergence du réseau de Kohonen
    carte_generee = []

    start_time = time.time()
    while entrainement:
        # On sauvegarde l'état du réseau avant l'entraînement
        neurones_avant_entraînement = neurones

        # Comptabilisation du nombre de victoire par neuronne
        nb_victoire_neurones = [0 for i in range(len(neurones))]
        # Mise à jour des poids
        for i in range(len(data.iloc[0, :])):
            epoch += 1
            # Recherche du neurone le plus proche de la ville d'index i
            ville = data.iloc[:, i]
            # Sauvegarde des caractéristiques du neurone le plus proche trouvé
            index, poids = le_plus_proche(neurones, ville)
            # Incrémentation du nombre de victoire du neurone index
            nb_victoire_neurones[index] += 1

            # Mise à jour de ses poids
            neurones[index] = maj_poids(poids, ville, epoch)

            # Mise à jour des poids de ses voisins direct
            # Cas où il y a 2 voisins
            if index < len(neurones)-1 and index >= 1:
                neurones[(index-1) % len(neurones)] = maj_poids_voisin(
                    neurones[index], neurones[index-1], ville, epoch)
                neurones[index+1 % len(neurones)] = maj_poids_voisin(neurones[index],
                                                                     neurones[index+1], ville, epoch)

        representation_itineraire_back(data, neurones)
        carte_generee.append(neurones)

        # Regroupement des neurones
        suppression = [i for i in range(
            0, len(neurones)) if nb_victoire_neurones[i] == 0]
        if len(suppression) > 0:
            suppression.sort()
            suppression.reverse()
            # Parcours des index des neurones ayant jamais gagné
            for i in suppression:
                del nb_victoire_neurones[i]
                del neurones[i]

        # Duplication des neurones qui gagnent beaucoup
        for i in range(len(neurones)):
            if nb_victoire_neurones[i] >= 2 * min(nb_victoire_neurones):
                neurones.insert(i, ajout_neurone(neurones[i]))

    temps_calcul = time.time() - start_time

    return carte_generee, temps_calcul


def main(data, matrice_distance, chemin_optimal=[]):
    """Lancement de l'algorithme de Kohonen

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : list
        matrice stockant l'integralité des distances inter villes
    chemin_optimal : list
        résulat optimal donné par la TSPLIB

    Returns
    -------
    dict
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    """

    if chemin_optimal != []:
        distance_chemin_optimal = distance_trajet(
            chemin_optimal, matrice_distance)

    resolution = {
        'Nombre de villes': len(chemin_optimal),
        'Algorithme': 'Réseau de neurones de Kohonen',
        'Distance': 'Euclidienne-2D',
        # Stockage de l'ensemble des cartes auto-génératrice
        'Chemins': [],
        'Chemin optimal': chemin_optimal,
        # Erreur par rapport à la solution optimal de la TSPLIB
        'Erreur (en %)': 0,
        'Temps de calcul (en s)': 0
    }

    x_barycentre, y_barycentre = barycentre(data)
    # Neurones initialment crées
    carte_initiale = initialisation_neurones(x_barycentre, y_barycentre, N)
    cartes_trouvees, temps_calcul = kohonen(data, carte_initiale)

    """
    # Ajout des chemins explorés au dictionnaire retourné
    resolution['Chemins'].extend(cartes_trouvees)
    # Calcul de la distance du trajet final trouvé par l'algorithme. En dernière position
    # de la variable précédente

    distance_chemin_sub_optimal = distance_trajet(
       resolution['Chemins'][-1], matrice_distance)
    # Calcul de l'erreur si un chemin optimal est renseigné
    if chemin_optimal != []:
        erreur = 100*(distance_chemin_sub_optimal -
                      distance_chemin_optimal)/distance_chemin_optimal
        resolution['Erreur (en %)'] = erreur
    """

    resolution['Temps de calcul (en s)'] = temps_calcul
    return resolution
