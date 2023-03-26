import pandas as pd

import src.algo_2_opt
import src.algo_genetique
import src.algo_proche_voisin
import src.algo_kohonen
from src.distance import matrice_distance
from src.init_test_data import data_TSPLIB
from src.affichage_resultats import affichage

# Nom des data de test
ENSEMBLE_TEST = ['dj38', 'xqf131', 'qa194', 'xqg237',
                 'pma343', 'pka379', 'pbl395', 'pbk411', 'pbn423']

# Nom des algo implémentés
ENSEMBLE_ALGOS = ['2_opt', 'plus_proche_voisin', 'genetique', 'kohonen']


def test_global(algo: str) -> pd.DataFrame:
    """Lancement de tous les tests unitaires pour un algorithme

    Parameters
    ----------
    algo : str
        le nom de l'algorithme à utiliser

    Returns
    -------
    Dataframe 
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme. Un ligne représente un test sur un jeu de données
    """
    assert algo in ENSEMBLE_ALGOS, print(
        "Veuillez choisir un algorithme parmi : {}".format(ENSEMBLE_ALGOS))

    # Dataframe à retourner, une ligne représente un test de l'algorithme
    df_resultat_test = pd.DataFrame({
        'Algorithme': [],
        'Nombre de villes': [],
        'Solution': [],
        # Distance du trajet final
        'Distance': [],
        'Temps de calcul (en s)': []
    })

    # Test sur l'ensemble des data
    for num_dataset in range(len(ENSEMBLE_TEST)):
        # Feeback d'avancement
        print(f"Etape du test : {num_dataset+1}/{len(ENSEMBLE_TEST)}")
        if algo == '2_opt':
            df_res = test_unitaire(num_dataset, algo)
        elif algo == 'plus_proche_voisin':
            df_res = test_unitaire(num_dataset, algo)
        elif algo == 'genetique':
            df_res = test_unitaire(num_dataset, algo)
        else:
            df_res = test_unitaire(num_dataset, algo)
        df_resultat_test = pd.concat(
            [df_resultat_test, df_res], ignore_index=True)  # type: ignore

    return df_resultat_test


def test_unitaire(num_dataset: int, algo: str) -> pd.DataFrame:
    """Lancement d'un test unitaire pour un algorithme

    Parameters
    ----------
    num_dataset : int
        numéro du dataset sur lequel est réalisé le test. Ce numéro est égale à son 
        index dans ENSEMBLE_TEST
    algo : str
        le nom de l'algorithme à utiliser

    Returns
    -------
    Dataframe
        variable stockant un ensemble de données importantes pour analyser
        l'algorithme
    """
    assert algo in ENSEMBLE_ALGOS, print(
        "Veuillez choisir un algorithme parmi : {}".format(ENSEMBLE_ALGOS))

    # Initialisation du dataframe avec TSPLIB
    data = data_TSPLIB(f'data/{ENSEMBLE_TEST[num_dataset]}.tsp')

    # Initialisation de la matrice des distances relatives
    mat_distance = matrice_distance(data)

    if algo == '2_opt':
        # On prend un chemin initial meilleur qu'un chemin aléatoire
        # Attention chemin_initial est la liste des chemins explorés par l'algorithme
        # plus_proche_voisin
        chemin_initial, temps_calcul = src.algo_proche_voisin.plus_proche_voisin(
            mat_distance)

        # Lancement de l'algorithme 2-opt
        df_res = src.algo_2_opt.main(mat_distance, chemin_initial)
        # Affichage du chemin trouvé et sauvegarde de la figure
        affichage(df_res, data, f'{algo}/chemin_{ENSEMBLE_TEST[num_dataset]}')
    elif algo == 'plus_proche_voisin':
        # Lancement de l'algorithme plus proche voisin
        df_res = src.algo_proche_voisin.main(mat_distance)
        # Affichage du chemin trouvé et sauvegarde de la figure
        affichage(df_res, data,
                  f'{algo}/chemin_{ENSEMBLE_TEST[num_dataset]}')
    elif algo == 'genetique':
        # Lancement de l'algorithme génétique
        df_res = src.algo_genetique.main(data, mat_distance)
        # Affichage du chemin trouvé et sauvegarde de la figure
        affichage(df_res, data,
                  f'{algo}/chemin_{ENSEMBLE_TEST[num_dataset]}')
    else:
        # Lancement de l'algorithme génétique
        df_res = src.algo_kohonen.main(data, mat_distance)
        # Affichage du chemin trouvé et sauvegarde de la figure
        affichage(df_res, data, f'{algo}/chemin_{ENSEMBLE_TEST[num_dataset]}')

    return df_res
