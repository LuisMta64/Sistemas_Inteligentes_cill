import numpy as np
import math

# def fitness_evaluation( n_allels, n_genes, scale, offset, x ):
#     decimal = binary2decimal( n_allels, n_genes, scale, offset, x )
#     aproxArea = abs(( 12.5 ) - ( decimal[0] * decimal[1] / 2 ))
#     aproxPerimetro =  abs((17.071067811) - ( decimal[0] + decimal[1] + math.sqrt( decimal[0]**2 + decimal[1]**2 ) ))
#     return [np.array( (aproxArea + aproxPerimetro) ), decimal ]


def fitness_evaluation (n_alleles, n_genes, scale, offset, x):
    x = binary2decimal(n_alleles, n_genes, scale, offset, x)
    fitness = abs(0.25-np.sin(x))
    return fitness

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