# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 10:31:59 2025

@author: tabti fatiha
"""
import pandas as pd
import numpy as np

# Exercice 1: Création et modification d'une Series
print("Exercice 1:")
# Création d'une série Pandas à partir d'un dictionnaire
series = pd.Series({'a': 100, 'b': 200, 'c': 300})
print("Série Pandas:")
print(series)

# Exercice 2: Création de DataFrames
print("Exercice 2:")
# Création d'un DataFrame avec des données
data = {'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]}
df = pd.DataFrame(data)
print("DataFrame original:")
print(df)
df['D'] = [10, 11, 12] # Ajout d'une nouvelle colonne D
print("DataFrame après ajout de la colonne D:")
print(df)
df = df.drop(columns=['B']) # Suppression de la colonne B
print("DataFrame après suppression de la colonne B:")
print(df)

# Exercice 3: Indexation et sélection dans un DataFrame
print("Exercice 3:")
data = {'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]}
df = pd.DataFrame(data)
column_B = df['B']  # Sélection de la colonne B
print("Sélection de la colonne B:")
print(column_B)
columns_AC = df[['A', 'C']]  # Sélection des colonnes A et C
print("Sélection des colonnes A et C:")
print(columns_AC)
row_1 = df.loc[1]  # Sélection de la ligne avec l'index 1
print("Sélection de la ligne avec l'index 1:")
print(row_1)

# Exercice 4: Ajout et suppression d'éléments dans un DataFrame
print("Exercice 4:")
df['Somme'] = df.sum(axis=1)  # Ajout d'une colonne Somme
print("DataFrame après ajout de la colonne Somme:")
print(df)
df = df.drop(columns=['Somme'])  # Suppression de la colonne Somme
print("DataFrame après suppression de la colonne Somme:")
print(df)
df['Aléatoire'] = np.random.rand(len(df))  # Ajout d'une colonne avec valeurs aléatoires
print("DataFrame après ajout de la colonne Aléatoire:")
print(df)

# Exercice 5: Fusion de DataFrames
print("Exercice 5:")
# Création de deux DataFrames avec une clé commune
left = pd.DataFrame({'key': [1, 2, 3], 'A': ['A1', 'A2', 'A3'], 'B': ['B1', 'B2', 'B3']})
right = pd.DataFrame({'key': [1, 2, 3], 'C': ['C1', 'C2', 'C3'], 'D': ['D1', 'D2', 'D3']})
# Fusion des DataFrames avec une jointure interne
merged_df = pd.merge(left, right, on='key')
print("DataFrame fusionné (jointure interne):")
print(merged_df)
# Fusion avec une jointure externe
merged_outer = pd.merge(left, right, on='key', how='outer')
print("DataFrame fusionné (jointure externe):")
print(merged_outer)
# Ajout d'une nouvelle colonne E et mise à jour de la fusion
right['E'] = ['E1', 'E2', 'E3']
merged_with_E = pd.merge(left, right, on='key', how='outer')
print("DataFrame fusionné avec la nouvelle colonne E:")
print(merged_with_E)

# Exercice 6: Nettoyage des données
print("Exercice 6:")
# Création d'un DataFrame contenant des valeurs NaN
data = {'A': [1.0, np.nan, 7.0], 'B': [np.nan, 5.0, 8.0], 'C': [3.0, 6.0, np.nan]}
df = pd.DataFrame(data)
df_filled = df.fillna(0)  # Remplacement des NaN par 0
print("DataFrame après remplacement des NaN par 0:")
print(df_filled)
df_mean_filled = df.fillna(df.mean())  # Remplacement des NaN par la moyenne de la colonne
print("DataFrame après remplacement des NaN par la moyenne de la colonne:")
print(df_mean_filled)
df_dropped = df.dropna()  # Suppression des lignes avec NaN
print("DataFrame après suppression des lignes contenant des NaN:")
print(df_dropped)

# Exercice 7: Groupement et agrégation
print("Exercice 7:")
# Création d'un DataFrame avec des catégories
data = {'Catégorie': ['A', 'B', 'A', 'B', 'A', 'B'], 'Valeur': [1, 2, 3, 4, 5, 6]}
df = pd.DataFrame(data)
# Calcul de la moyenne des valeurs par catégorie
grouped_mean = df.groupby('Catégorie')['Valeur'].mean()
print("Regroupé par catégorie - Moyenne des valeurs:")
print(grouped_mean)
# Calcul de la somme des valeurs par catégorie
grouped_sum = df.groupby('Catégorie')['Valeur'].sum()
print("Regroupé par catégorie - Somme des valeurs:")
print(grouped_sum)
# Comptage des entrées par catégorie
grouped_count = df.groupby('Catégorie')['Valeur'].count()
print("Regroupé par catégorie - Nombre d'entrées:")
print(grouped_count)

# Exercice 8: Tableaux croisés dynamiques (Pivot Tables)
print("Exercice 8:")
# Création d'un DataFrame avec une catégorie, un type et une valeur associée
data = {
    'Catégorie': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Type': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Valeur': [1, 2, 3, 4, 5, 6]
}
df = pd.DataFrame(data)
# Création d'un tableau croisé avec la moyenne des valeurs par catégorie et type
pivot_mean = df.pivot_table(values='Valeur', index='Catégorie', columns='Type', aggfunc='mean')
print("Tableau croisé - Moyenne des valeurs:")
print(pivot_mean)
# Modification du tableau pour afficher la somme au lieu de la moyenne
pivot_sum = df.pivot_table(values='Valeur', index='Catégorie', columns='Type', aggfunc='sum')
print("Tableau croisé - Somme des valeurs:")
print(pivot_sum)
# Ajout des marges pour afficher le total global
pivot_margins = df.pivot_table(values='Valeur', index='Catégorie', columns='Type', aggfunc='sum', margins=True)
print("Tableau croisé avec marges:")
print(pivot_margins)

# Exercice 9: Séries temporelles
print("Exercice 9:")
# Création d'une série temporelle avec une plage de dates
date_range = pd.date_range(start='2023-01-01', periods=6, freq='D')
df = pd.DataFrame({'Date': date_range, 'Valeur': np.random.randint(1, 100, 6)})
df.set_index('Date', inplace=True)
# Regroupement des données par périodes de 2 jours et calcul de la somme
resampled = df.resample('2D').sum()
print("Séries temporelles - Somme des périodes de 2 jours:")
print(resampled)

# Exercice 10: Gestion des valeurs manquantes
print("Exercice 10:")
# Création d'un DataFrame contenant des valeurs manquantes
data = {'A': [1.0, 2.0, np.nan], 'B': [np.nan, 5.0, 8.0], 'C': [3.0, np.nan, 9.0]}
df = pd.DataFrame(data)
# Interpolation des valeurs manquantes
df_interpolated = df.interpolate()
print("Données après interpolation des valeurs manquantes:")
print(df_interpolated)
# Suppression des lignes contenant des valeurs manquantes
df_dropped_na = df.dropna()
print("Données après suppression des lignes avec valeurs manquantes:")
print(df_dropped_na)

import warnings
warnings.filterwarnings('ignore') # supprimer les warnings 
# Exercice 11: Opérations sur un DataFrame
print("Exercice 11:")
# Création d'un DataFrame avec des valeurs numériques
data = {'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]}
df = pd.DataFrame(data)
# Calcul de la somme cumulative des valeurs
df_cumsum = df.cumsum()
print("Somme cumulative:")
print(df_cumsum)
# Calcul du produit cumulatif
df_cumprod = df.cumprod()
print("Produit cumulatif:")
print(df_cumprod)
# Application d'une fonction qui soustrait 1 à toutes les valeurs du DataFrame
df_subtract = df.applymap(lambda x: x - 1)
print("DataFrame après soustraction de 1:")
print(df_subtract)
