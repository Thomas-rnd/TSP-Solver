import matplotlib.pyplot as plt
import pandas as pd


def representation_itineraire(data):
    """Affichage des N villes par des points ainsi que le parcours réalisé
       Le parcours est donné par l'ordre des villes dans le dataframe"""
    # Affichage des points
    plt.scatter(data.iloc[0, :], data.iloc[1, :], zorder=1)
    # Repérage du point initial par un cercle rouge
    plt.scatter(data.iloc[0, 0], data.iloc[1, 0], zorder=1,
                color="red", marker='o', label='Point de Départ')
    plt.legend(loc="upper right")

    # Affichage des traits
    plt.plot(data.iloc[0, :], data.iloc[1, :], zorder=1)
    plt.title('Chemin parcouru par le marchand', loc='center',)
    # Pour une visualisation plus proche de la réalité
    # plt.axis("equal")
    plt.show()
