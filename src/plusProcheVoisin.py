from testData import trajet_en_df
from graph import representation_itineraire


def plus_proche_voisin(data, matrice_distance):
    """Retourne le trajet trouvé en se déplacement de proche en proche.
    La ville de départ étant arbitraire je choisis la ville d'index 0"""
    itineraire = [0]
    while len(data.loc['x']) != len(itineraire):
        # A chaque itération on cherche la ville la plus proche de la ville actuelle
        # la ville actuelle étant la dernière de l'itinéraire

        # Liste trié dans l'ordre croissant des distances entre la ville actuelle et le reste
        distances = sorted(matrice_distance[itineraire[-1]])

        # On enlève la distance qui correspond à rester sur la même ville
        distances.remove(0)

        i = 0
        # On recherche la ville la plus proche encore inexplorée
        while matrice_distance[itineraire[-1]].index(distances[i]) in itineraire:
            i += 1
        itineraire.append(
            matrice_distance[itineraire[-1]].index(distances[i]))
    itineraire.append(itineraire[0])
    return itineraire
