from github import pygame_simulation

pygame_simulation(tour=2, dist_ratio=4, fs=(1.5, 1, 0.75, 0.5, 0.25),
                  max_iter=12000, alpha=0.99, beta=0.90)
