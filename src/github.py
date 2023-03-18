import random
import time
import math

from distance import distance_euclidienne, distance_trajet
from init_test_data import data_TSPLIB
from distance import matrice_distance


def initialisation_neurones(data, nb=3):
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
    if nb == 0:
        nb = len(data.iloc[0, :])

    # coordonnées maximale
    max_x = max(data.iloc[0, :]) / 2
    max_y = max(data.iloc[1, :]) / 2

    if nb > 1:
        # on dispose les neurones en ellipse
        n = []
        for i in range(0, nb):
            x = max_x + max_x * math.cos(math.pi * 2 * float(i) / nb) / 4
            y = max_y + max_y * math.sin(math.pi * 2 * float(i) / nb) / 4
            n.append((x, y))
        return n
    else:
        n = [(max_x, max_y)]
        return n


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


def poids_attirance(poids, dist):
    """
    Calcule le poids d'attraction d'une neurone vers une ville.
    """
    d = distance_euclidienne(poids[0], poids[0], poids[1], poids[1])
    d = dist / (d + dist)
    return d


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


def deplace_neurone(n, villes, neurones, dist_w, forces, compte):
    """
    Déplace le neurone le plus proche de la ville *n*,
    déplace ses voisins.

    @param    villes        liste des villes
    @param    neurones      liste des neurones
    @param    dist          distance d'attirance
    @param    forces        force de déplacement des voisins du neurones
    @param    compte        incrémente compte [n] où n est l'indice du neurone choisi
    @return                 indice du neurone le plus proche
    """
    # recherche du neurone le plus proche
    index, poids = le_plus_proche(neurones, villes[n])

    # vecteur de déplacement
    compte[index] += 1
    n = neurones[index]
    vec = soustraction_vecteurs(poids, n)
    poids = poids_attirance(vec, dist_w)
    vec = multiplier_un_vecteur(vec, poids)
    n = addition_vecteurs(n, vec)
    neurones[index] = n

    # déplacement des voisins
    for k in range(0, len(forces)):
        i1 = (index + k + 1) % len(neurones)
        i2 = (index - k - 1 + len(neurones)) % len(neurones)
        n1 = neurones[i1]
        n2 = neurones[i2]

        vec = soustraction_vecteurs(n, n1)
        poids = poids_attirance(vec, dist_w)
        vec = multiplier_un_vecteur(vec, poids)
        vec = multiplier_un_vecteur(vec, forces[k])
        n1 = addition_vecteurs(n1, vec)

        vec = soustraction_vecteurs(n, n2)
        poids = poids_attirance(vec, dist_w)
        vec = multiplier_un_vecteur(vec, poids)
        vec = multiplier_un_vecteur(vec, forces[k])
        n2 = addition_vecteurs(n2, vec)

        neurones[i1] = n1
        neurones[i2] = n2

    return index


def iteration(villes, neurones, dist, forces, compte_v, compte_n):
    """
    Choisit une ville aléatoirement et attire le neurones le plus proche,
    choisit cette ville parmi les villes les moins fréquemment choisies.

    @param    villes     liste des villes
    @param    neurones   liste des neurones
    @param    dist       distance d'attirance
    @param    forces     force de déplacement des voisins du neurones
    @param    compte_v   incrémente compte_v [n] où n est l'indice de la ville choisie
    @param    compte_n   incrémente compte_n [n] où n est l'indice du neurone choisi
    @return              indices de la ville et du neurone le plus proche
    """
    m = min(compte_v)
    ind = [i for i in range(0, len(villes)) if compte_v[i] == m]
    n = random.randint(0, len(ind) - 1)
    n = ind[n]
    compte_v[n] += 1
    return n, deplace_neurone(n, villes, neurones, dist, forces, compte_n)


def modifie_structure(neurones, compte, nb_sel):
    """
    Modifie la structure des neurones, supprime les neurones jamais
    déplacés, et ajoute des neurones lorsque certains sont trop sollicités.
    """
    def cmp_add(i, j):
        return -1 if i[0] < j[0] else (1 if i[0] > j[0] else 0)

    if nb_sel > 0:
        # supprime les neurones les moins sollicités
        sup = [i for i in range(0, len(neurones)) if compte[i] == 0]
        if len(sup) > 0:
            sup.sort()
            sup.reverse()
            for i in sup:
                del compte[i]
                del neurones[i]

        # on ajoute un neurone lorsque max (compte) >= 2 * min (compte)
        add = []
        for i in range(0, len(compte)):
            if compte[i] > nb_sel:
                d1 = math.sqrt(distance_euclidienne(neurones[i],
                                                    neurones[(i + 1) % len(neurones)]))
                d2 = math.sqrt(distance_euclidienne(neurones[i],
                                                    neurones[(i - 1 + len(neurones)) % len(neurones)]))
                if d1 > d2:
                    d1 = d2
                p = neurones[i]
                p = (p[0] + random.randint(0, int(d1 / 2)),
                     p[1] + random.randint(0, int(d1 / 2)))
                add.append((i, p, 0))

        add = list(sorted(add, key=functools.cmp_to_key(cmp_add)))
        add.reverse()
        for a in add:
            neurones.insert(a[0], a[1])
            compte.insert(a[0], a[2])

    # on remet les compteurs à zéros
    for i in range(0, len(compte)):
        compte[i] = 0


def pygame_simulation(tour=2, dist_ratio=4, fs=(1.5, 1, 0.75, 0.5, 0.25),
                      max_iter=12000, alpha=0.99, beta=0.90):
    """
    @param      pygame          module pygame
    @param      first_click     attend la pression d'un clic de souris avant de commencer
    @param      folder          répertoire où stocker les images de la simulation
    @param      size            taille de l'écran
    @param      delay           delay between two tries
    @param      flags           see `pygame.display.set_mode <https://www.pygame.org/docs/ref/display.html#pygame.display.set_mode>`_
    @param      fLOG            logging function
    @param      fs              paramètres
    @param      max_iter        nombre d'itérations maximum
    @param      alpha           paramètre alpha
    @param      beta            paramètre beta
    @param      dist_ratio      ratio distance
    @param      tour            nombre de tours
    @param      nb              nombre de points

    La simulation ressemble à ceci :

    .. raw:: html

        <video autoplay="" controls="" loop="" height="125">
        <source src="http://www.xavierdupre.fr/enseignement/complements/tsp_kohonen.mp4" type="video/mp4" />
        </video>

    Pour lancer la simulation::

        from ensae_teaching_cs.special.tsp_kohonen import pygame_simulation
        import pygame
        pygame_simulation(pygame)

    Voir :ref:`l-puzzle_girafe`.
    """
    data = data_TSPLIB()
    villes = [ville for ville in data.iloc[:]]
    # Initialisation de la matrice des distances relatives#
    mat_distance = matrice_distance(data)
    # Neurones initialment crées

    neurones = initialisation_neurones(data, 3)
    compte_n = [0 for i in neurones]
    compte_v = [0 for i in data.iloc[0, :]]
    maj = tour * len(data.iloc[0, :])
    # Attention on ajoute deux fois les mêmes distances il est symétrique
    somme_distance = [sum(dist) for dist in mat_distance]
    somme_distance = sum(somme_distance)
    moyenne_distance_inter_ville = somme_distance / \
        (len(mat_distance)*len(mat_distance)*2)
    dist = moyenne_distance_inter_ville * dist_ratio

    iter = 0
    while iter < max_iter:
        iter += 1

        if iter % maj == 0:
            modifie_structure(neurones, compte_n, tour)
            dist *= alpha
            f2 = tuple(w * beta for w in fs)
            fs = f2

        bv, bn = iteration(villes, neurones, dist, fs, compte_v, compte_n)
        representation_itineraire(data, neurones)
