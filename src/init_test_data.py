import numpy as np
import pandas as pd


# Pour favoriser la réutilisation par la comunauté scientifique
# les tests des algorithmes implémentés peuvent être réalisées sur
# les données officielles de ce problème.

# Lien de téléchargement des fichier .tsp
# http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/


def data_TSPLIB(fichier: str) -> pd.DataFrame:
    """
    Lecture d'un fichier au format .tsp en copiant les informations dans 
    un dataframe pandas

    Parameters
    ----------
    fichier : str
        nom du fichier à traiter. Fichier dans le dossier `data`

    Returns
    -------
    DataFrame
        L'ensemble des villes ainsi crées depuis le fichier .tsp. Sous la forme 
        `'Ville', 'x', 'y'`
    """
    with open(fichier) as f:
        noeud_coord_debut = None
        dimension = None
        lignes = f.readlines()

        # Lecture des informations du fichier .tsp
        i = 0
        while not dimension or not noeud_coord_debut:
            ligne = lignes[i]
            if ligne.startswith('DIMENSION :'):
                dimension = int(ligne.split()[-1])
            if ligne.startswith('NODE_COORD_SECTION'):
                noeud_coord_debut = i
            i = i+1

        # On définit le point de référence au début du fichier
        f.seek(0)

        villes = pd.read_csv(
            f,
            # On commence la lecture du fichier au bon endroit
            skiprows=noeud_coord_debut + 1,
            # Définition du séparateur
            sep=' ',
            # Définition des colonnes du dataframe
            names=['Ville', 'x', 'y'],
            # Définition du type des colonnes
            dtype={'Ville': str, 'x': np.float64, 'y': np.float64},
            header=None,
            nrows=dimension
        )

        return villes


def trajet_en_df(trajet: list[int], data: pd.DataFrame) -> pd.DataFrame:
    """Convertion d'un trajet en un dataframe afin de l'afficher simplement

    Parameters
    ----------
    trajet : list
        list ordonne de villes
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir

    Returns
    -------
    DataFrame
        DataFrame ordonné pour afficher correctement le trajet trouvé
    """
    # Récupération des coordonnées des villes pour pouvoir les afficher
    x = []
    y = []
    index = []
    for i in trajet:
        x.append(data.iloc[i, 1])
        y.append(data.iloc[i, 2])
        index.append(i)
    # Un dataframe d'une ligne par ville
    df_res = pd.DataFrame({'Ville': index, 'x': x, 'y': y})
    return df_res


def normalisation(villes: pd.DataFrame) -> pd.Series:
    """Normalisation des coordonnées des villes afin de faciliter
    l'apprentissage du réseau de neuronnes

    Parameters
    ----------
    villes : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
        sans la colonne `Ville`

    Returns
    -------
    Series
        Villes du dataframe normalisées
    """
    ratio = (villes.x.max() - villes.x.min()) / \
        (villes.y.max() - villes.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = villes.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)
