import numpy as np
import math

def fitness_evaluation(n_alleles, n_genes, scale, offset, chromosome):
    x = binary2decimal(n_alleles, n_genes, scale, offset, chromosome)
    
    try:
        A = 10
        fitness = A * n_genes + sum(xi**2 - A * np.cos(2 * math.pi * xi) for xi in x)
        fitness = 1 / (1 + abs(fitness))
    except:
        fitness = 0
    
    return fitness, x

def binary2decimal(n_alleles, n_genes, scale, offset, chromosome):
    list_values = []
    for i in range(n_genes):
        value = 0
        for j in range(n_alleles):
            bit_index = i * n_alleles + j
            value += chromosome[bit_index] * 2**(n_alleles - 1 - j)
        decimal = value / (2**n_alleles - 1) * scale + offset
        list_values.append(decimal)
    return np.array(list_values)