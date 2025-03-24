# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 16:08:44 2025

@author: tabti
"""
#Exercice 1:
import numpy as np
import matplotlib.pyplot as plt
# Création des données x et y pour la fonction polynomiale
x = np.linspace(-10, 10, 100)
y = 2 * x**3 - 5 * x**2 + 3 * x - 7
# Tracé de la fonction polynomiale
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label=r'$y = 2x^3 - 5x^2 + 3x - 7$')
# Ajout des labels et du titre
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of the Polynomial Function')
plt.legend()
# Affichage du graphique 
plt.grid(True)
plt.show()

#Exercice 2:
import numpy as np
import matplotlib.pyplot as plt
# Création des données x et y pour les fonctions exponentielle et logarithmique
x = np.linspace(0.1, 10, 500)
y1 = np.exp(x)  # Fonction exponentielle
y2 = np.log(x)  # Fonction logarithmique
# Tracé des deux fonctions sur le même graphique
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'r-', label=r'$y = exp(x)$')
plt.plot(x, y2, 'g--', label=r'$y = log(x)$')
# Personnalisation du graphique
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential and Logarithmic Functions')
plt.legend()
plt.grid(True)
# Sauvegarde du graphique
plt.savefig('exp_log_plot.png', dpi=100)
plt.show()

#Exercice 3:
import numpy as np
import matplotlib.pyplot as plt
# Création d'un tableau x pour les différentes fonctions
x1 = np.linspace(-2*np.pi, 2*np.pi, 500)  # Pour tan et arctan
x2 = np.linspace(-2, 2, 500)  # Pour sinh et cosh
# Calcul des fonctions
y_tan = np.tan(x1)
y_arctan = np.arctan(x1)
y_sinh = np.sinh(x2)
y_cosh = np.cosh(x2)
# Création de la figure avec deux sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
# Premier subplot : tan et arctan
axes[0].plot(x1, y_tan, color="blue", linewidth=2, linestyle="-", label=r"$y = \tan(x)$")
axes[0].plot(x1, y_arctan, color="red", linewidth=2, linestyle="--", label=r"$y = \arctan(x)$")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(r"Fonctions $\tan(x)$ et $\arctan(x)$")
axes[0].legend()
axes[0].grid(True)
axes[0].set_ylim(-10, 10)  # Limite de l'axe y pour éviter les valeurs infinies de tan(x)
# Deuxième subplot : sinh et cosh
axes[1].plot(x2, y_sinh, color="green", linewidth=2, linestyle="-", label=r"$y = \sinh(x)$")
axes[1].plot(x2, y_cosh, color="purple", linewidth=2, linestyle="--", label=r"$y = \cosh(x)$")
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(r"Fonctions $\sinh(x)$ et $\cosh(x)$")
axes[1].legend()
axes[1].grid(True)
# Ajustement de la mise en page
fig.tight_layout()
# Affichage du graphique
plt.show()


# Génération d'un échantillon de 1000 valeurs suivant une distribution normale
n = np.random.randn(1000)
# Création de la figure pour l'histogramme
plt.figure(figsize=(8, 6), dpi=100)
# Tracé de l'histogramme avec 30 bins et une couleur personnalisée
plt.hist(n, bins=30, color="skyblue", edgecolor="black")
# Ajout des labels et du titre
plt.xlabel('Valeurs')
plt.ylabel('Fréquence')
plt.title("Histogramme de 1000 valeurs issues d'une distribution normale")
# Définition des limites de l'axe x
plt.xlim(np.min(n), np.max(n))
# Affichage du graphique
plt.show()
   
#Exercice 5:
import numpy as np
import matplotlib.pyplot as plt
# Création de données aléatoires pour x et y
x = np.random.uniform(-5, 5, 500)
y = np.sin(x) + np.random.normal(0, 0.5, 500)
# Tracé du scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=y, s=20, cmap='viridis', edgecolors='k')
# Personnalisation du graphique
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot with Mathematical Function')
plt.grid(True)
# Sauvegarde du graphique
plt.savefig('scatter_plot.pdf')
plt.show()

#Exercice 6:
import numpy as np
import matplotlib.pyplot as plt
# Création des données pour un plot de contour
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
# Tracé du contour plot
plt.figure(figsize=(10, 6))
cp = plt.contour(X, Y, Z, cmap='viridis')
# Ajout du titre et des labels
plt.title('Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp)
plt.show()

#Exercice 7:
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Création des données pour le graphique 3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
# Création du graphique 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
# Ajout des labels et du titre
ax.set_title('3D Surface Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#Exercice 8:
import numpy as np
import matplotlib.pyplot as plt
# Création de l'array x avec 10 points espacés linéairement entre -2 et 2
x = np.linspace(-2, 2, 10)
# Définition des fonctions y1, y2 et y3
y1 = x**2
y2 = x**3
y3 = x**4
# Tracé des trois fonctions avec différentes couleurs, styles de ligne et marqueurs
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'r--o', label=r'$y = x^2$')  # Ligne rouge, tirets, cercles comme marqueurs
plt.plot(x, y2, 'g-^', label=r'$y = x^3$')  # Ligne verte, traits continus, triangles comme marqueurs
plt.plot(x, y3, 'b:*', label=r'$y = x^4$')  # Ligne bleue, points discontinus, étoiles comme marqueurs
# Personnalisation du graphique
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of Different Functions with Different Line and Marker Styles')
plt.legend()
# Affichage du graphique avec la grille
plt.grid(True)
plt.show()

#Exercice 9:
import numpy as np
import matplotlib.pyplot as plt
# Création de l'array x avec 50 points espacés linéairement entre 1 et 100
x = np.linspace(1, 100, 50)
# Calcul des fonctions y1 et y2
y1 = 2**x
y2 = np.log2(x)
# Tracé des deux fonctions avec échelle logarithmique pour l'axe y
plt.figure(figsize=(12, 6))
plt.plot(x, y1, label=r'$y = 2^x$', color='blue')
plt.plot(x, y2, label=r'$y = log_2(x)$', color='red')
# Personnalisation du graphique
plt.yscale('log')  # Échelle logarithmique pour l'axe y
plt.xlabel('x')
plt.ylabel('y')
plt.title('Logarithmic Scale for y-axis')
plt.legend()
plt.grid(True)
# Affichage du graphique
plt.show()

#Exercice 10:
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Création des données pour le graphique 3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
# Création de la figure
fig = plt.figure(figsize=(12, 6))
# Création du premier subplot avec un angle de vue initial
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('View Angle 1: (azim=45, elev=30)')
ax1.view_init(azim=45, elev=30)
# Création du deuxième subplot avec un angle de vue différent
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis')
ax2.set_title('View Angle 2: (azim=-45, elev=60)')
ax2.view_init(azim=-45, elev=60)
# Affichage du graphique avec deux sous-graphes
plt.tight_layout()
plt.show()

#Exercice 11:
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Création des données pour X et Y avec un pas de 0.25
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
# Création des grilles X et Y
X, Y = np.meshgrid(x, y)
# Calcul de Z selon la formule donnée
Z = np.sin(X) * np.cos(Y)
# Création de la figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Tracé du graphique wireframe
ax.plot_wireframe(X, Y, Z, color='black')
# Personnalisation
ax.set_title('3D Wireframe Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Changer l'angle de vue
ax.view_init(azim=45, elev=30)
# Affichage du graphique
plt.show()

#Exercice 12:   
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Création des données pour X et Y avec un pas de 0.25
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
# Création des grilles X et Y
X, Y = np.meshgrid(x, y)
# Calcul de Z selon la formule donnée
Z = np.exp(-0.1 * (X**2 + Y**2))
# Création de la figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Tracé du graphique de contours en 3D
ax.contour3D(X, Y, Z, 50, cmap='viridis')
# Personnalisation
ax.set_title('3D Contour Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Ajouter une barre de couleur
plt.colorbar(ax.contour3D(X, Y, Z, 50, cmap='viridis'))
# Affichage du graphique
plt.show()
  
#Exercice 13:
# Création de l'array t avec 100 points entre 0 et 2π
t = np.linspace(0, 2*np.pi, 100)
# Calcul des coordonnées paramétriques
X = np.sin(t)
Y = np.cos(t)
Z = t
# Création de la figure et du subplot 3D
fig = plt.figure(figsize=(10, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')
# Tracé du graphe paramétrique avec une couleur personnalisée (bleu)
ax.plot(X, Y, Z, color="blue", linewidth=2)
# Ajout des labels des axes
ax.set_xlabel("X = sin(t)")
ax.set_ylabel("Y = cos(t)")
ax.set_zlabel("Z = t")
ax.set_title("Tracé paramétrique 3D de $(X, Y, Z) = (\sin(t), \cos(t), t)$")
# Affichage du graphique
plt.show()
    
#Exercice 14:    
# Création des tableaux x et y avec 10 points entre -5 et 5
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)
# Aplatir les tableaux pour une utilisation avec bar3d
X_flat = X.flatten()
Y_flat = Y.flatten()
# Calcul des valeurs de Z
Z = np.exp(-0.1 * (X_flat**2 + Y_flat**2))
# Largeur et profondeur des barres
dx = dy = 0.5  # Taille des barres
# Création de la figure et du subplot 3D
fig = plt.figure(figsize=(10, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')
# Définition de la normalisation des couleurs
norm = plt.Normalize(vmin=Z.min(), vmax=Z.max())
colors = plt.cm.coolwarm(norm(Z))  # Application de la colormap
# Tracé du bar plot en 3D avec couleurs adaptées aux valeurs de Z
bars = ax.bar3d(X_flat, Y_flat, np.zeros_like(Z), dx, dy, Z, color=colors, shade=True)
# Création d'un ScalarMappable pour la barre de couleur
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])  # Nécessaire pour éviter l'erreur
# Ajout d'une barre de couleur avec l'axe spécifié
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label("Valeurs de Z")
# Changement de l'angle de vue
ax.view_init(elev=30, azim=45)
# Ajout des labels des axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Graphique en barres 3D de $Z = e^{-0.1 (X^2 + Y^2)}$")
# Affichage du graphique
plt.show()
    
#Exercice 15:
# Création des tableaux X, Y et Z avec des valeurs entre -5 et 5
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
z = np.linspace(-5, 5, 5)
X, Y, Z = np.meshgrid(x, y, z)
# Calcul des composantes du champ vectoriel
U = -Y  # Composante en X
V = X   # Composante en Y
W = Z   # Composante en Z
# Création de la figure et du subplot 3D
fig = plt.figure(figsize=(10, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')
# Tracé du champ vectoriel avec une couleur et une échelle ajustée
ax.quiver(X, Y, Z, U, V, W, length=0.5, color='blue', normalize=True)
# Ajout des labels des axes et du titre
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Champ vectoriel 3D")
# Affichage du graphique
plt.show()
    
#Exercice 16:  
import numpy as np
import matplotlib.pyplot as plt
# Génération des données aléatoires
np.random.seed(42)  # Pour obtenir des résultats reproductibles
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)
# Création de la figure et du subplot 3D
fig = plt.figure(figsize=(10, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')
# Tracé du nuage de points 3D
scatter_plot = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
# Ajout d'une barre de couleur
color_bar = plt.colorbar(scatter_plot, ax=ax, shrink=0.5, aspect=10)
color_bar.set_label("Valeurs de Z")
# Ajout des labels des axes
ax.set_xlabel("Axe X")
ax.set_ylabel("Axe Y")
ax.set_zlabel("Axe Z")
# Ajout d'un titre
ax.set_title("Nuage de Points 3D avec Coloration par Z")
# Affichage du graphique
plt.show()
 
#Exercice 17:  
# Génération des données
t = np.linspace(0, 4 * np.pi, 100)
X = np.sin(t)
Y = np.cos(t)
Z = t
# Création de la figure et du subplot 3D
fig = plt.figure(figsize=(10, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')
# Tracé de la ligne 3D
ax.plot(X, Y, Z, color='red', linewidth=2.5)
# Ajout des labels des axes
ax.set_xlabel("X = sin(t)")
ax.set_ylabel("Y = cos(t)")
ax.set_zlabel("Z = t")
# Ajout d'un titre
ax.set_title("Tracé d'une Ligne 3D Paramétrique")
# Affichage du graphique
plt.show()
 
#Exercice 18: 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Création des données pour X et Y avec un pas de 0.1
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
# Création des grilles X et Y
X, Y = np.meshgrid(x, y)
# Calcul de Z selon la formule donnée
Z = np.sin(np.sqrt(X**2 + Y**2))
# Création de la figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Tracé du contour plot 3D rempli
ax.contourf(X, Y, Z, 20, cmap='viridis')
# Personnalisation
ax.set_title('3D Filled Contour Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Affichage du graphique
plt.show()
 
#Exercice 19: 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Création des données pour X et Y avec 50 points entre -5 et 5
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
# Création des grilles X et Y
X, Y = np.meshgrid(x, y)
# Calcul de Z selon la formule donnée
Z = np.sin(np.sqrt(X**2 + Y**2))
# Création de la figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Tracé du graphique de la heatmap 3D
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
# Personnalisation
ax.set_title('3D Heatmap')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Ajout de la barre de couleur
fig.colorbar(surf)
# Affichage du graphique
plt.show()

#Exercice 20:
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Générer 1000 points aléatoires dans l'espace 3D
x = np.random.randn(1000)
y = np.random.randn(1000)
z = np.random.randn(1000)
# Création de la figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Tracé du graphique de densité 3D avec des points
ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', alpha=0.5)
# Personnalisation
ax.set_title('3D Density Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Affichage du graphique
plt.show()

    
    
    