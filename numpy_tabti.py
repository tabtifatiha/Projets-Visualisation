# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:28:32 2025

@author: tabti fatiha
"""

#Exercices
import numpy as np
#Exercise 1: Creating and Manipulating NumPy Arrays
print("Exercise 1:")
array_1d = np.array([5, 10, 15, 20, 25], dtype=np.float64)
print("1D Array:", array_1d)
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Shape:", array_2d.shape, "Size:", array_2d.size)
array_3d = np.random.rand(2, 3, 4)
print("3D Shape:", array_3d.shape, "Dimensions:", array_3d.ndim)

#Exercise 2: Advanced Array Manipulations
print("Exercise 2:")
array_1d_reversed = np.arange(10)[::-1]
print("Reversed 1D:", array_1d_reversed)
array_2d_shape = np.arange(12).reshape(3, 4)
subarray = array_2d_shape[:2, -2:]
print("Subarray:", subarray)
array_5x5 = np.random.randint(0, 10, (5, 5))
array_5x5[array_5x5 > 5] = 0
print("Modified 5x5:", array_5x5)

#Exercise 3: Array Initialization and Attributes
print("Exercise 3:")
identity_matrix = np.eye(3)
print("Identity Matrix:", identity_matrix)
print(identity_matrix.shape," / ", identity_matrix.size," / ", identity_matrix.ndim, " / ", identity_matrix.itemsize," / ", identity_matrix.nbytes)
evenly_spaced = np.linspace(0, 5, 10)
evenly_spaced_dtype = evenly_spaced.dtype
print("Evenly Spaced:", evenly_spaced, "Dtype:", evenly_spaced_dtype)
array_3d_random = np.random.randn(2, 3, 4)
sum_elements = np.sum(array_3d_random)
print("Sum of 3D Array:", sum_elements)

#Exercise 4: Fancy Indexing and Masking
print("Exercise 4:")
array_random = np.random.randint(0, 50, 20)
selected_elements = array_random[[2, 5, 7, 10, 15]]
print("Selected Elements:", selected_elements)
array_2d_mask = np.random.randint(0, 30, (4, 5))
masked_elements = array_2d_mask[array_2d_mask > 15]
print("Masked Elements:", masked_elements)
array_1d_negative = np.random.randint(-10, 10, 10)
array_1d_negative[array_1d_negative < 0] = 0
print("Modified 1D:", array_1d_negative)

#Exercise 5: Combining and Splitting Arrays
print("Exercise 5:")
array1 = np.random.randint(0, 10, 5)
array2 = np.random.randint(0, 10, 5)
concatenated = np.concatenate((array1, array2))
print("Concatenated:", concatenated)
array_2d_split = np.random.randint(0, 10, (6, 4))
split_arrays = np.split(array_2d_split, 2, axis=0)
print("Split Arrays:", split_arrays)
array_2d_column_split = np.random.randint(0, 10, (3, 6))
column_splits = np.split(array_2d_column_split, 3, axis=1)
print("Column Splits:", column_splits)

#Exercise 6: Mathematical Functions and Aggregations
print("Exercise 6:")
array_stat = np.random.randint(1, 100, 15)
mean_value = np.mean(array_stat)
median_value = np.median(array_stat)
std_dev = np.std(array_stat)
variance = np.var(array_stat)
print("Mean:", mean_value, "Median:", median_value, "Std Dev:", std_dev, "Variance:", variance)
array_2d = np.random.randint(1, 50, (4, 4))
row_sums = np.sum(array_2d, axis=1)
col_sums = np.sum(array_2d, axis=0)
print(row_sums , col_sums)
array_3d = np.random.randint(1, 20, (2, 3, 4))
max_along_axes = np.max(array_3d, axis=(0, 1, 2))
min_along_axes = np.min(array_3d, axis=(0, 1, 2))
print( max_along_axes , min_along_axes)

#Exercise 7: Reshaping and Transposing Arrays
print("Exercise 7:")
array_reshape = np.arange(1, 13).reshape(3, 4)
print("Reshaped Array:", array_reshape)
array_transpose = np.random.randint(1, 10, (3, 4)).T
print("Transposed Array:", array_transpose)
array_flatten = np.random.randint(1, 10, (2, 3)).flatten()
print("Flattened Array:", array_flatten)

#Exercise 8: Broadcasting and Vectorized Operations
print("Exercise 8:")
array_broadcast = np.random.randint(1, 10, (3, 4))
column_mean = array_broadcast.mean(axis=0)
normalized_array = array_broadcast - column_mean
print("Normalized Array:", normalized_array)
array1 = np.random.randint(1, 5, 4)
array2 = np.random.randint(1, 5, 4)
outer_product = np.outer(array1, array2)
print("Outer Product:", outer_product)
array_large = np.random.randint(1, 10, (4, 5))
array_large[array_large > 5] += 10
print("Large Array:", array_large)

#Exercise 9: Sorting and Searching Arrays
print("Exercise 9:")
array_sort = np.random.randint(1, 20, 10)
sorted_array = np.sort(array_sort)
print(sorted_array)
array_2d_sort = np.random.randint(1, 50, (3, 5))
array_2d_sorted = array_2d_sort[array_2d_sort[:, 1].argsort()]
print( array_2d_sorted)
array_search = np.random.randint(1, 100, 15)
indices = np.where(array_search > 50)
print(indices)
print("Valeurs correspondantes:", array_search[indices])

#Exercise 10: Linear Algebra with NumPy
print("Exercise 10:")
matrix_A = np.random.randint(1, 10, (2, 2))
determinant = np.linalg.det(matrix_A)
print(determinant)
matrix_B = np.random.randint(1, 5, (3, 3))
eigenvalues, eigenvectors = np.linalg.eig(matrix_B)
print(eigenvalues)
matrix_C = np.random.randint(1, 10, (2, 3))
matrix_D = np.random.randint(1, 10, (3, 2))
matrix_product = np.dot(matrix_C, matrix_D)
print(matrix_product)

#Exercise 11: Random Sampling and Distributions
print("Exercise 11:")
import matplotlib.pyplot as plt
uniform_sample = np.random.uniform(0, 1, 10)
print(uniform_sample)
normal_sample = np.random.normal(0, 1, (3, 3))
print(normal_sample)
random_ints = np.random.randint(1, 100, 20)
plt.hist(random_ints, bins=5, edgecolor="black")
plt.show()

#Exercise 12: Advanced Indexing and Selection
print("Exercise 12:")
# Sélection des éléments diagonaux
array_2d = np.random.randint(1, 21, (5, 5))
print(np.diagonal(array_2d))
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
array_1d = np.random.randint(1, 51, 10)
primes = array_1d[np.vectorize(is_prime)(array_1d)]
print(primes)
array_2d_even = np.random.randint(1, 11, (4, 4))
even_numbers = array_2d_even[array_2d_even % 2 == 0]
print(even_numbers)

#Exercise 13: Handling Missing Data
import numpy as np
print("Exercise 13:")
array_nan = np.random.randint(1, 11, 10).astype(float)
indices = np.random.choice(len(array_nan), size=3, replace=False)
array_nan[indices] = np.nan
print(array_nan)
array_2d_nan = np.random.randint(1, 11, (3, 4)).astype(float)
array_2d_nan[array_2d_nan < 5] = np.nan
print(array_2d_nan)
array_1d_15 = np.random.randint(1, 21, 15).astype(float)
nan_indices = np.random.choice(len(array_1d_15), size=4, replace=False) 
array_1d_15[nan_indices] = np.nan
indices_nans = np.where(np.isnan(array_1d_15))[0] 
print("Tableau avec NaN:", array_1d_15)
print("Indices des valeurs NaN:", indices_nans)

#Exercise 14: Performance Optimization with NumPy
print("Exercise 14:")
import time
large_array = np.random.randint(1, 101, 10**6)
start = time.time()
mean_val = np.mean(large_array)
std_dev = np.std(large_array)
end = time.time()
print(f"Moyenne: {mean_val}, Ecart-type: {std_dev}, Temps: {end - start:.5f} sec")
matrix_A = np.random.randint(1, 11, (1000, 1000))
matrix_B = np.random.randint(1, 11, (1000, 1000))
start = time.time()
matrix_sum = matrix_A + matrix_B
end = time.time()
print(f"Temps d'exécution de l'addition: {end - start:.5f} sec")
array_3d = np.random.randint(1, 11, (100, 100, 100))
start = time.time()
sum_axis0 = np.sum(array_3d, axis=0)
sum_axis1 = np.sum(array_3d, axis=1)
sum_axis2 = np.sum(array_3d, axis=2)
end = time.time()
print(f"Temps de calcul des sommes: {end - start:.5f} sec")

#Exercise 15: Cumulative and Aggregate Functions
print("Exercise 15:")
array_1d = np.arange(1, 11)
cumsum_array = np.cumsum(array_1d)
cumprod_array = np.cumprod(array_1d)
print("Somme cumulative:", cumsum_array)
print("Produit cumulatif:", cumprod_array)
array_2d = np.random.randint(1, 21, (4, 4))
cumsum_rows = np.cumsum(array_2d, axis=1)
cumsum_cols = np.cumsum(array_2d, axis=0)
print("Somme cumulative par ligne:", cumsum_rows)
print("Somme cumulative par colonne:", cumsum_cols)
array_random = np.random.randint(1, 51, 10)
print("Minimum:", np.min(array_random))
print("Maximum:", np.max(array_random))
print("Somme:", np.sum(array_random))

#Exercise 16: Working with Dates and Times
print("Exercise 16:")
dates_daily = np.arange(np.datetime64('today'), np.datetime64('today') + 10, dtype='datetime64[D]')
print("Dates journalières:", dates_daily)
dates_monthly = np.arange('2022-01', '2022-06', dtype='datetime64[M]')
print("Dates mensuelles:", dates_monthly)
random_days = np.random.randint(0, 365, 10)
timestamps_2023 = np.datetime64('2023-01-01') + random_days
print("Timestamps 2023:", timestamps_2023)

#Exercise 17: Creating Arrays with Custom Data Types
print("Exercise 17:")
dtype_custom = np.dtype([('nombre', np.int32), ('binaire', 'U10')])
array_binary = np.array([(i, bin(i)) for i in range(5)], dtype=dtype_custom)
print("Tableau avec représentation binaire:", array_binary)
dtype_complex = np.dtype([('valeur', np.complex128)])
array_complex = np.array([[complex(1, 2), complex(3, 4), complex(5, 6)],
                          [complex(7, 8), complex(9, 10), complex(11, 12)],
                          [complex(13, 14), complex(15, 16), complex(17, 18)]], dtype=dtype_complex)
print("Tableau de nombres complexes:", array_complex)
dtype_books = np.dtype([('Titre', 'U50'), ('Auteur', 'U50'), ('Pages', np.int32)])
books = np.array([("Le Petit Prince", "Antoine de Saint-Exupéry", 96),
                  ("1984", "George Orwell", 328),
                  ("Les Misérables", "Victor Hugo", 1232)], dtype=dtype_books)
print("Tableau structuré des livres:", books)


















