import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px

from init_test_data import trajet_en_df


def representation_itineraire_back(data, reseau_neurones=[]):
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


def representation_itineraire_web(data):
    """Affichage des N villes par des points ainsi que le parcours réalisé
       Le parcours est donné par l'ordre des villes dans le dataframe

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des informations sur un algorithme

    Returns
    -------
    fig
        Graphique de visualisation plolty
    """
    fig = px.line(data, x='x', y='y',
                  title='Chemin parcouru par le marchand', markers=True)
    return fig


def representation_temps_calcul(fichier_csv):
    """Affichage des du temps de calcul d'un algorithme en fonction
    du nombre de ville qu'il a traité

    Parameters
    ----------
    fichier_csv : str
        fichier csv stockant l'intégralité des résultats des différents algorithmes

    Returns
    -------
    fig
        Graphique de visualisation plolty
    """
    data = pd.read_csv(fichier_csv)
    data['ln(Temps de calcul (en s))']=np.log(data['Temps de calcul (en s)'])
    #fig = px.scatter(data, x='Nombre de villes',
    #              y='ln(Temps de calcul (en s))', color='Algorithme',
    #              title='Représentation du temps de calcul en fonction du nombre de ville à explorer', trendline="ols")
    fig = px.line(data, x='Nombre de villes',
                  y='ln(Temps de calcul (en s))', color='Algorithme',
                  title='Représentation du temps de calcul en fonction du nombre de ville à explorer',markers=True)
    fig.write_image("../resultats/figures/fig_temps_calcul.svg")
    return fig


def representation_resultats(fichier_csv):
    """Affichage des N villes par des points ainsi que le parcours réalisé
       Le parcours est donné par l'ordre des villes dans le dataframe

    Parameters
    ----------
    data : DataFrame
        Dataframe stockant l'intégralité des informations sur un algorithme

    Returns
    -------
    fig
        Graphique de visualisation plolty
    """
    data = pd.read_csv(fichier_csv)
    fig = px.box(data, x="Algorithme", y="Distance",color="Algorithme",
             title="Distance du chemin trouvé en fonction de l'algorithme"
            )
    fig.write_image("../resultats/figures/fig_distances.svg")
    return fig


def affichage(df_resolution, data, nom_fichier):
    """Affichage d'un trajet et des performances d'un algorithme
    Parameters
    ----------
    df_resolution : Dataframe
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    Returns
    -------
    fig
        Graphique de visualisation plolty
    """
    df_meilleur_trajet = trajet_en_df(
        df_resolution['Solution'][0], data)
    # fig = representation_itineraire(df_meilleur_trajet)
    fig = representation_itineraire_web(df_meilleur_trajet)
    fig.write_image(f"../resultats/figures/{nom_fichier}.svg")

    print("=============================================")
    print("Nombre de ville : ", df_resolution["Nombre de villes"][0])
    print("Distance : ", df_resolution["Distance"][0])
    print("Temps de calcul (en s): ",
          df_resolution["Temps de calcul (en s)"][0])
    print("=============================================")

    return fig
