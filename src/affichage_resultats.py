import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from src.init_test_data import data_TSPLIB, normalisation, trajet_en_df

# Chemin de stockage des différents fichiers numériques
ROOT = "resultats/figures/"


def representation_itineraire_web(data: pd.DataFrame, chemins: pd.DataFrame, nom_fichier="") -> go.Figure:
    """Affichage des N villes par des points ainsi que le parcours réalisé
       Le parcours est donné par l'ordre des villes dans le dataframe

    Parameters
    ----------
    data : DataFrame
        dataframe stockant l'intégralité des coordonnées des villes à parcourir

    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    # Affichage des villes
    fig = px.scatter(data, x='x', y='y', template="simple_white",
                     title="Shortest path found by the algorithm")

    # On relie les villes dans le bon ordre
    fig.add_trace(
        go.Scatter(
            x=chemins['x'].values,
            y=chemins['y'].values,
            mode='lines',
            showlegend=False)

    )
    fig.update_xaxes(zeroline=False, visible=False)
    fig.update_yaxes(zeroline=False, visible=False)

    # Sauvegarde de la figure au format .png
    if (nom_fichier != ""):
        fig.write_image(nom_fichier)
    return fig


def representation_reseau(data: pd.DataFrame, neurones: np.ndarray, nom_fichier="") -> go.Figure:
    """Affichage des N villes par des points ainsi que la projection du réseaux
    de neurones sur l'espace des villes

    Parameters
    ----------
    data : DataFrame
        dataframe stockant l'intégralité des coordonnées des villes à parcourir (villes normalisées)
    reseau_neurones : np.ndarray
        vecteur stockant un réseau de neurone de kohonen

    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    # Affichage des villes
    fig = px.scatter(data, x='x', y='y', template="simple_white",
                     title="Organisation of the Kohonen neurons network")

    # On relie les neurones dans le bon ordre
    fig.add_trace(
        go.Scatter(
            x=[neurone[0] for neurone in neurones],
            y=[neurone[1] for neurone in neurones],
            mode='lines+markers',
            showlegend=False)
    )
    fig.update_xaxes(zeroline=False, visible=False)
    fig.update_yaxes(zeroline=False, visible=False)

    # Sauvegarde de la figure au format .png
    if (nom_fichier != ""):
        fig.write_image(nom_fichier)
    return fig


def representation_temps_calcul(fichier_csv: str) -> go.Figure:
    """Affichage du temps de calcul des différents algorithmes implémentés
    en fonction du nombre de villes à parcourir

    Parameters
    ----------
    fichier_csv : str
        fichier csv stockant l'intégralité des résultats des différents algorithmes

    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    # Lecture du fichier stockant l'ensemble des résultats
    data = pd.read_csv(fichier_csv)
    # fig = px.scatter(data, x='Nombre de villes',
    #              y='ln(Temps de calcul (en s))', color='Algorithme',
    #              title='Représentation du temps de calcul en fonction du nombre de ville à explorer', trendline="ols")
    fig = px.line(data, x='Nombre de villes',
                  y='Temps de calcul (en s)', color='Algorithme', template='plotly_white',
                  title='Representation of the calculation time according to the number of cities to explore', markers=True, log_y=True,
                  labels={"Nombre de villes": "Number of cities", "Temps de calcul (en s)": "Calculation time (in s)",
                          "Algorithme": 'Algorithm'})

    newnames = {'2-opt': '2-opt inversion', 'plus_proche_voisin': 'Nearest neighbor search',
                'genetique': 'Genetic algorithm', 'kohonen': 'Kohonen algorithm'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(
        t.name, newnames[t.name])
    ))
    # Sauvegarde de la figure au format .png
    fig.write_image("resultats/figures/fig_temps_calcul.png")
    return fig


def representation_resultats(fichier_csv: str) -> go.Figure:
    """Affichage des distances des chemins trouvés par algorithme

    Parameters
    ----------
    fichier_csv : str
        fichier csv stockant l'intégralité des résultats des différents algorithmes

    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    # Lecture du fichier stockant l'ensemble des résultats
    data = pd.read_csv(fichier_csv)

    fig = px.box(data, x="Algorithme", y="Distance", color="Algorithme", template='plotly_white',
                 title="Distance of the path found according to the algorithm", points="all",
                 category_orders={'Algorithme': ['2-opt inversion', 'Nearest neighbor search',
                                                 'Genetic algorithm', 'Kohonen algorithm']},
                 labels={"Nombre de villes": "Number of cities", "Temps de calcul (en s)": "Calculation time (in s)",
                         "Algorithme": 'Algorithm', "Génétique": 'Genetic'}
                 )

    newnames = {'2-opt': '2-opt inversion', 'plus_proche_voisin': 'Nearest neighbor search',
                'genetique': 'Genetic algorithm', 'kohonen': 'Kohonen algorithm'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(
        t.name, newnames[t.name])
    )
    )
    # Sauvegarde de la figure au format .png
    fig.write_image("resultats/figures/fig_distances.png")
    return fig


def affichage(df_resolution: pd.DataFrame, data: pd.DataFrame, nom_fichier="") -> go.Figure:
    """Affichage d'un trajet et des performances d'un algorithme

    Parameters
    ----------
    df_resolution : Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    data : DataFrame
        dataframe stockant l'intégralité des coordonnées des villes à parcourir
    nom_fichier : str (optionnel)
        nom du fichier si on souhaite sauvegarder la figure crée

    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    # Création d'un dataframe complet issu de la solution trouvée
    df_meilleur_trajet = trajet_en_df(
        df_resolution['Solution'][0], data)
    if nom_fichier != "":
        fig = representation_itineraire_web(
            data, df_meilleur_trajet, f"{ROOT+nom_fichier}.png")
    else:
        fig = representation_itineraire_web(
            data, df_meilleur_trajet)

    # Affichage console de certain résultat
    # print("=============================================")
    # print("Nombre de ville : ", df_resolution["Nombre de villes"][0])
    # print("Distance : ", df_resolution["Distance"][0])
    # print("Temps de calcul (en s): ",
    #      df_resolution["Temps de calcul (en s)"][0])
    # print("=============================================")
    return fig


def affichage_chemins_explores(exploration: list[list[int]], algorithme: str, dataset: str):
    """Sauvegarde des chemins explorés au format `.png` par un algorithme 

    Parameters
    ----------
    exploration : list[list[int]]
        variable retraçant la méthode d'exploration de l'algorithme
    algorithme : str
        nom de l'algorithme à traiter
    dataset : str 
        nom du dataset à traiter
    """
    # On récupère la liste des chemins explorés de ligne associé au bon algorithme et au bon dataset
    # Si stocké dans un dataframe
    # chemins_explores = df_resolution.loc[(df_resolution["Algorithme"] == algorithme) & (
    #   df_resolution["Nom dataset"] == dataset)]["Chemins explorés"]

    # On génère le dataframe associé à ce dataset
    data = data_TSPLIB(f'data/{dataset}.tsp')
    # Pour chaque chemin exploré nous allons sauvegarder la figure associée
    for index, chemin in enumerate(exploration):
        df_meilleur_trajet = trajet_en_df(chemin, data)
        fig = representation_itineraire_web(data, df_meilleur_trajet)
        # Sauvegarde de la figure au format .png
        fig.write_image("{}{}/{}/{:05d}.png".format(ROOT,
                                                    algorithme, dataset, index))


def affichage_reseau_neurones(exploration: list[np.ndarray], algorithme: str, dataset: str):
    """Sauvegarde de l'évolution du réseau de neurones 

    Parameters
    ----------
    exploration : list[np.ndarray]
        variable retraçant l'évolution du réseau de neurones
    algorithme : str
        nom de l'algorithme à traiter
    dataset : str 
        nom du dataset à traiter
    """
    # On récupère la liste des chemins explorés de ligne associé au bon algorithme et au bon dataset
    # Si stocké dans un dataframe
    # chemins_explores = df_resolution.loc[(df_resolution["Algorithme"] == algorithme) & (
    #   df_resolution["Nom dataset"] == dataset)]["Chemins explorés"]

    # On génère le dataframe associé à ce dataset
    data = data_TSPLIB(f'data/{dataset}.tsp')
    # On crée des villes artificielles normalisées pour être en cohérence avec le domain des poids des neurones [0,1]
    villes = data.copy()
    villes[['x', 'y']] = normalisation(villes[['x', 'y']])
    # Pour chaque chemin exploré nous allons sauvegarder la figure associée
    for index, reseau in enumerate(exploration):
        fig = representation_reseau(villes, reseau)
        # Sauvegarde de la figure au format .png
        fig.write_image("{}{}/{}/{:05d}.png".format(ROOT,
                                                    algorithme, dataset, index))


def generation_gif(algorithme: str, dataset: str):
    """Génération du fichier `.gif` pour nous permettre de bien visualiser le fonctionnemnt 
    d'un algorithme

    Parameters
    ----------
    algorithme : str
        nom de l'algorithme à traiter
    dataset : str 
        nom du dataset à traiter
    """
    # Récupération de toutes les images du dossier de manière ordonné
    dossier_images = '{}/{}/'.format(algorithme, dataset)
    images = [Image.open("{}{}/{}".format(ROOT, dossier_images, image))
              for image in sorted(os.listdir(ROOT+dossier_images))]
    premiere_image = images[0]
    premiere_image.save(f"gif/{algorithme}_{dataset}.gif", format="GIF", append_images=images,
                        save_all=True, duration=200, loop=0)
    # Suppression des images indépendantes pour ne pas saturer le disque
    for image in glob.glob(f"{ROOT+dossier_images}/*.png"):
        os.remove(image)
    # Feedback de la création du gif
    print(
        f"Vous venez de créer le fichier '{algorithme}_{dataset}.gif' dans le dossier gif à la racine du projet")
