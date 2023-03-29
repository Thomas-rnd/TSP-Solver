import numpy as np
import pandas as pd

import src.algo_2_opt
import src.algo_genetique
import src.algo_kohonen
import src.algo_proche_voisin
from src.affichage_resultats import affichage, affichage_chemins_explores
from src.distance import matrice_distance
from src.init_test_data import data_TSPLIB

# Nom des data de test
ENSEMBLE_TEST = ['dj38', 'xqf131', 'qa194', 'xqg237',
                 'pma343', 'pka379', 'pbl395', 'pbk411', 'pbn423']

# Nom des algo implémentés
ENSEMBLE_ALGOS = ['2-opt', 'plus_proche_voisin', 'genetique', 'kohonen']


def test_global(algorithme: str) -> pd.DataFrame:
    """Lancement de tous les tests unitaires pour un algorithme

    Parameters
    ----------
    algo : str
        le nom de l'algorithme à utiliser parmi `['2-opt', 'plus_proche_voisin', 'genetique', 'kohonen']`

    Returns
    -------
    Dataframe 
        Données retourné sur l'algorithme : 
        `'Algorithme', 'Nom dataset', 'Nombre de villes', 'Solution', 'Distance', 'Temps de calcul (en s)'`
    """
    assert algorithme in ENSEMBLE_ALGOS, print(
        "Veuillez choisir un algorithme parmi : {}".format(ENSEMBLE_ALGOS))

    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        # Nom de l'algorithme
        'Algorithme': [],
        # Nom du jeu de données
        'Nom dataset': [],
        # Nombre de villes du jeu de données
        'Nombre de villes': [],
        # Chemin le plus court trouvé
        'Solution': [],
        # Distance du trajet final
        'Distance': [],
        # Temps mis par l'algorithme pour trouver le résultat
        'Temps de calcul (en s)': []
    })
    # Test sur l'ensemble des jeux de données
    for num_dataset in range(len(ENSEMBLE_TEST)):
        # Feeback d'avancement
        print(f"Etape du test : {num_dataset+1}/{len(ENSEMBLE_TEST)}")
        if algorithme == '2-opt':
            df_res, _ = test_unitaire(num_dataset, algorithme)
        elif algorithme == 'plus_proche_voisin':
            df_res, _ = test_unitaire(num_dataset, algorithme)
        elif algorithme == 'genetique':
            df_res, _ = test_unitaire(num_dataset, algorithme)
        else:
            df_res, _ = test_unitaire(num_dataset, algorithme)
        df_resultat_test = pd.concat(
            [df_resultat_test, df_res], ignore_index=True)

    return df_resultat_test


def test_unitaire(num_dataset: int, algo: str) -> tuple[pd.DataFrame, list[list[int]] | list[np.ndarray]]:
    """Lancement d'un test unitaire pour un algorithme

    Parameters
    ----------
    num_dataset : int
        numéro du dataset sur lequel est réalisé le test. Ce numéro est égale à son 
        index dans `ENSEMBLE_TEST`
    algo : str
        le nom de l'algorithme à utiliser

    Returns
    -------
    df_res : Dataframe
        Données retourné sur l'algorithme sur un jeu de données : 
        `'Algorithme', 'Nom dataset', 'Nombre de villes', 'Solution', 'Distance', 'Temps de calcul (en s)'`
    exploration : list[list[int]] | list[np.ndarray]
        variable stockant l'évolution de la recherche de l'algorithme
    """
    assert algo in ENSEMBLE_ALGOS, print(
        "Veuillez choisir un algorithme parmi : {}".format(ENSEMBLE_ALGOS))

    # Initialisation du dataframe avec TSPLIB
    data = data_TSPLIB(f'data/{ENSEMBLE_TEST[num_dataset]}.tsp')

    # Initialisation de la matrice des distances relatives
    mat_distance = matrice_distance(data)

    if algo == '2-opt':
        # On prend un chemin initial meilleur qu'un chemin aléatoire
        chemin_initial, _, _ = src.algo_proche_voisin.plus_proche_voisin(
            mat_distance)
        # Lancement de l'algorithme 2-opt
        df_res, exploration = src.algo_2_opt.main(
            mat_distance, chemin_initial, ENSEMBLE_TEST[num_dataset])

    elif algo == 'plus_proche_voisin':
        # Lancement de l'algorithme plus proche voisin
        df_res, exploration = src.algo_proche_voisin.main(
            mat_distance, ENSEMBLE_TEST[num_dataset])

    elif algo == 'genetique':
        # Lancement de l'algorithme génétique
        df_res = src.algo_genetique.main(
            data, mat_distance, ENSEMBLE_TEST[num_dataset])
        # Il n'y a pas de réelle méthode d'exploration : le fruit du hasard
        # Exploration est donc initialisée à une liste vide pour conserver le bon type
        exploration = []

    else:
        # Lancement de l'algorithme de kohonen
        df_res, exploration = src.algo_kohonen.main(
            data, mat_distance, ENSEMBLE_TEST[num_dataset])

    return df_res, exploration
