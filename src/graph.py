import matplotlib.pyplot as plt
import pandas as pd

from testData import trajet_en_df


def representation_itineraire(data, reseau_neurones=[]):
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


def affichage(resolution, data):
    """Affichage d'un trajet et des performances d'un algorithme

    Parameters
    ----------
    resolution : dict
        variable stockant un ensemble de variables importantes pour analyser
        l'algorithme
    data : DataFrame
        Dataframe stockant l'intégralité des coordonnées des villes à parcourir
    """
    df_meilleur_trajet = trajet_en_df(resolution['Chemins'][-1], data)
    representation_itineraire(df_meilleur_trajet)

    print("Analyse de la performance de l'algorithme :",
          resolution["Algorithme"])
    print("=============================================")
    print("Nombre de ville : ", resolution["Nombre de villes"])
    print("Pourcentage d'erreur : ", resolution["Erreur (en %)"])
    print("Temps de calcul (en s): ", resolution["Temps de calcul (en s)"])
    print("=============================================")
