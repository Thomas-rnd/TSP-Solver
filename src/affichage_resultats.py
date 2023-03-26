import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.init_test_data import trajet_en_df


def representation_itineraire_web(data: pd.DataFrame) -> go.Figure:
    """Affichage des N villes par des points ainsi que le parcours réalisé
       Le parcours est donné par l'ordre des villes dans le dataframe

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir

    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    fig = px.scatter(data, x='x', y='y', template="simple_white",
                     title="Shortest path found by the algorithm")
    fig.add_trace(
        go.Scatter(
            x=data['x'].values,
            y=data['y'].values,
            mode='lines',
            showlegend=False)

    )
    fig.update_xaxes(zeroline=False, visible=False)
    fig.update_yaxes(zeroline=False, visible=False)
    return fig


def representation_reseau(data: pd.DataFrame, neurones: np.ndarray) -> go.Figure:
    """Affichage des N villes par des points ainsi que la projection du réseaux
    de neurones sur l'espace des villes

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    reseau_neurones : np.ndarray
        list stockant un réseau de neurone de kohonen

    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    fig = px.scatter(data, x='x', y='y', template="simple_white",
                     title="Organisation of the Kohonen neurons network")
    fig.add_trace(
        go.Scatter(
            x=[neurone[0] for neurone in neurones],
            y=[neurone[1] for neurone in neurones],
            mode='lines+markers',
            showlegend=False)
    )
    fig.update_xaxes(zeroline=False, visible=False)
    fig.update_yaxes(zeroline=False, visible=False)
    return fig


def representation_temps_calcul(fichier_csv: str) -> go.Figure:
    """Affichage des du temps de calcul des différents algorithmes implémentés
    en fonction du nombre de ville évalué

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

    newnames = {'2-opt': '2-opt inversion', 'Plus proche voisin': 'Nearest neighbor search',
                'Génétique': 'Genetic algorithm', 'Kohonen': 'Kohonen algorithm'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(
        t.name, newnames[t.name])
    ))
    # Sauvegarde de la figure au format .png
    # fig.write_image("resultats/figures/fig_temps_calcul.png")
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

    newnames = {'2-opt': '2-opt inversion', 'Plus proche voisin': 'Nearest neighbor search',
                'Génétique': 'Genetic algorithm', 'Kohonen': 'Kohonen algorithm'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(
        t.name, newnames[t.name])
    )
    )
    # Sauvegarde de la figure au format .png
    # fig.write_image("resultats/figures/fig_distances.png")
    return fig


def affichage(df_resolution: pd.DataFrame, data: pd.DataFrame, nom_fichier="") -> go.Figure:
    """Affichage d'un trajet et des performances d'un algorithme
    Parameters
    ----------
    df_resolution : Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    nom_fichier : str (optionnel)
        Nom du fichier si on souhaite sauvegarder la figure crée
    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    # Création d'un dataframe complet issu de la solution trouvée
    df_meilleur_trajet = trajet_en_df(
        df_resolution['Solution'][0], data)
    # fig = representation_itineraire(df_meilleur_trajet)
    fig = representation_itineraire_web(df_meilleur_trajet)
    # Sauvegarde de la figure au format .png
    if (nom_fichier != ""):
        fig.write_image(f"resultats/figures/{nom_fichier}.png")

    # Affichage console de certain résultat
    # print("=============================================")
    # print("Nombre de ville : ", df_resolution["Nombre de villes"][0])
    # print("Distance : ", df_resolution["Distance"][0])
    # print("Temps de calcul (en s): ",
    #      df_resolution["Temps de calcul (en s)"][0])
    # print("=============================================")

    return fig
