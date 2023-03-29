import time

import numpy as np
import pandas as pd

from src.distance import distance_trajet


# En s'inspirant de la documentation wikipedia sur le 2-opt pour résoudre le TSP, nous
# allons essayer de l'implémenter. Cet algorithme donne un résultat sub-optimal en temps
# très raisonnable. L'unique optimisation que nous réaliserons concerne la maj de la distance
# parcourue. En effet, on ne calculera uniquement la partie du trajet qui se verra modifiée
# à la suite de l'inversion. Cela concerne nous 2 arêtes du graph à parcourir.

# Cf. https://fr.wikipedia.org/wiki/2-opt


def gain(matrice_distance: np.ndarray, chemin_actuel: list[int], i: int, j: int) -> float:
    """Gain de distance en parcourant en sens inverse une suite de ville.

    On calcule la différence de distance entre la somme des anciennes arêtes et
    la somme des nouvelles arêtes formées. Si cette somme est positive on vient de trouver
    deux arêtes qui étaient sécantes avant l'inversion.

    Parameters
    ----------
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes
    chemin_actuel : list[int]
        suite de villes donnant le chemin parcouru
    i : int
        indice de la ville où commence l'inversion
    j : int
        indice de la ville où finie l'inversion

    Returns
    -------
    float
        le gain effectif de l'inversion
    """
    avant_permutation = chemin_actuel[i-1]
    debut_permutation = chemin_actuel[i]
    fin_permutation = chemin_actuel[j]
    apres_permuation = chemin_actuel[j+1]

    # Distance avant inversion
    distance_initiale = matrice_distance[avant_permutation, debut_permutation
                                         ]+matrice_distance[fin_permutation, apres_permuation]
    # Distance après inversion
    distance_finale = matrice_distance[avant_permutation, fin_permutation
                                       ]+matrice_distance[debut_permutation, apres_permuation]
    return distance_initiale-distance_finale


def inversion(liste: list, debut_inversion: int, fin_inversion: int) -> list:
    """Inversion d'une partie du chemin parcouru

    Parameters
    ----------
    liste : list
        une liste à renverser
    debut_inversion : int
        index de la ville où l'inversion commence
    fin_inversion : int
        index de la ville où l'inversion finie

    Returns
    -------
    list
        la liste renversée
    """
    liste_a_inverser = liste[debut_inversion:fin_inversion+1]
    nouvelle_liste = liste[:debut_inversion] + \
        liste_a_inverser[::-1] + liste[fin_inversion+1:]
    return nouvelle_liste


def deux_opt(itineraire_initial: list[int], matrice_distance: np.ndarray) -> tuple[list[int], float, list[list[int]]]:
    """Recherche des arêtes sécantes.

    Cette fonction implémente l'algorithme 2-opt décrit sur wikipédia.

    Parameters
    ----------
    itineraire_initial : list[int]
        suite de villes donnant le chemin parcouru. Ce chemin initial influ énormément
        sur le temps de calcul.
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes

    Returns
    -------
    chemin_explores : list[int]
        le chemin final trouvé
    temps_calcul : float
        temps necessaire à la résolution du problème
    chemins_explores : list[list[int]]
        stockage de l'ensemble des chemins explorés
    """
    start_time = time.time()

    # Variable d'arrêt de la recherche d'arêtes sécantes
    amelioration = True

    # Stockage du meilleur résultat courant
    meilleur_chemin = itineraire_initial
    meilleur_distance = distance_trajet(meilleur_chemin, matrice_distance)
    nombre_ville = len(meilleur_chemin)

    # Stockage de l'ensemble des trajets explorés
    chemins_explores = [meilleur_chemin]

    while amelioration:
        amelioration = False
        for debut_inversion in range(1, nombre_ville - 2):
            for fin_inversion in range(debut_inversion + 1, nombre_ville - 1):
                # Evaluation du gain de l'inversion
                if (gain(matrice_distance, meilleur_chemin, debut_inversion, fin_inversion)) > 0:
                    nouveau_chemin = inversion(
                        meilleur_chemin, debut_inversion, fin_inversion)
                    nouvelle_distance = distance_trajet(
                        nouveau_chemin, matrice_distance)

                    if (nouvelle_distance < meilleur_distance):
                        meilleur_chemin = nouveau_chemin
                        meilleur_distance = nouvelle_distance
                        chemins_explores.append(nouveau_chemin)
                        amelioration = True

    temps_calcul = time.time() - start_time
    return meilleur_chemin, temps_calcul, chemins_explores


def main(matrice_distance: np.ndarray, chemin_initial: list, nom_dataset="") -> tuple[pd.DataFrame, list[list[int]]]:
    """Lancement de l'algorithme de recherche

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    matrice_distance : np.ndarray
        matrice stockant l'integralité des distances inter villes
    nom_dataset : str (optionnel)
        Nom du dataset à traiter

    Returns
    -------
    df_resultat_test : Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    chemins_explores : list[list[int]]
        variable retraçant les chemins explorés par l'algorithme
    """
    # Résolution du TSP
    itineraire, temps_calcul, chemins_explores = deux_opt(
        chemin_initial, matrice_distance)

    # Calcul de la distance du trajet final trouvé par l'algorithme
    distance_chemin_sub_optimal = distance_trajet(itineraire, matrice_distance)

    # Création du dataframe à retourner
    # On inclut pas les chemins explorés pour pas sucharger le fichier csv de résultats
    df_resultat_test = pd.DataFrame({
        'Algorithme': "2-opt",
        'Nom dataset': nom_dataset,
        'Nombre de villes': len(chemin_initial)-1,
        # Dans un tableau pour être sur une seule ligne du dataframe
        'Solution': [itineraire],
        # Distance du trajet final
        'Distance': distance_chemin_sub_optimal,
        'Temps de calcul (en s)': temps_calcul
    })

    return df_resultat_test, chemins_explores
