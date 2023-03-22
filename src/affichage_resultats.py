import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.express as px

from src.init_test_data import trajet_en_df


def representation_itineraire_back(data: pd.DataFrame, reseau_neurones=[]):
    """Affichage des N villes par des points ainsi que le parcours réalisé
       Le parcours est donné par l'ordre des villes dans le dataframe

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    reseau_neurones : list
        list stockant un réseau de neurone de kohonen
    """
    # Affichage des points
    plt.scatter(data.iloc[0, :], data.iloc[1, :], zorder=1)
    # Repérage du point initial par un cercle rouge
    plt.scatter(data.iloc[0, 0], data.iloc[1, 0], zorder=1,
                color="red", marker='o', label='Point de Départ')
    plt.legend(loc="upper right")

    if (reseau_neurones == []):
        # Affichage des traits
        plt.plot(data.iloc[0, :], data.iloc[1, :], zorder=1)
        plt.title('Chemin parcouru par le marchand', loc='center')
        # Pour une visualisation plus proche de la réalité
        # plt.axis("equal")
    else:
        x = [neurone[0] for neurone in reseau_neurones]
        y = [neurone[1] for neurone in reseau_neurones]
        # Affichage des neurones
        plt.scatter(x, y, zorder=1,
                    color="green", marker='x', label='Réseau de Kohonen')
        # Affichage des traits
        plt.plot(x, y, zorder=1)
        plt.title('Chemin parcouru par le réseau', loc='center')
    plt.show()


def plot_network(cities, neurons, name='diagram.png', ax=None):
    """Plot a graphical representation of the problem"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        axis.plot(neurons[:, 0], neurons[:, 1], 'r.',
                  ls='-', color='#0063ba', markersize=2)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        ax.plot(neurons[:, 0], neurons[:, 1], 'r.',
                ls='-', color='#0063ba', markersize=2)
        return ax


def plot_route(cities, route, name='diagram.png', ax=None):
    """Plot a graphical representation of the route obtained"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        axis.plot(route['x'], route['y'], color='purple', linewidth=1)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    else:
        ax.scatter(cities['x'], cities['y'], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        ax.plot(route['x'], route['y'], color='purple', linewidth=1)
        return ax


def representation_itineraire_web(data: pd.DataFrame) -> px.line:
    """Affichage des N villes par des points ainsi que le parcours réalisé
       Le parcours est donné par l'ordre des villes dans le dataframe

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des informations sur un algorithme

    Returns
    -------
    Figure
        Graphique de visualisation plolty
    """
    fig = px.line(data, x='x', y='y',
                  title='Chemin parcouru par le marchand', markers=True)
    return fig


def representation_temps_calcul(fichier_csv: str) -> px.line:
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
    # Passange en échelle logarithmique
    data['ln(Temps de calcul (en s))'] = np.log(data['Temps de calcul (en s)'])
    # fig = px.scatter(data, x='Nombre de villes',
    #              y='ln(Temps de calcul (en s))', color='Algorithme',
    #              title='Représentation du temps de calcul en fonction du nombre de ville à explorer', trendline="ols")
    fig = px.line(data, x='Nombre de villes',
                  y='ln(Temps de calcul (en s))', color='Algorithme',
                  title='Représentation du temps de calcul en fonction du nombre de ville à explorer', markers=True)
    # Sauvegarde de la figure au format .png
    fig.write_image("resultats/figures/fig_temps_calcul.png")
    return fig


def representation_resultats(fichier_csv: str) -> px.box:
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
    fig = px.box(data, x="Algorithme", y="Distance", color="Algorithme",
                 title="Distance du chemin trouvé en fonction de l'algorithme"
                 )
    # Sauvegarde de la figure au format .png
    fig.write_image("resultats/figures/fig_distances.png")
    return fig


def affichage(df_resolution: pd.DataFrame, data: pd.DataFrame, nom_fichier="") -> px.line:
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
