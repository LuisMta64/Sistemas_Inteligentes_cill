import numpy as np

def binary2Decimal(chromosome, n_genes, n_alleles, scale, offset):
   list_values = []
   for i in range(n_genes):
        value = 0
        for j in range(n_alleles):
            bit_index = i * n_alleles + j
            value += chromosome[bit_index] * 2**(n_alleles - 1 - j)
        decimal = value / (2**n_alleles - 1) * scale + offset
        list_values.append(decimal)
   return np.array(list_values)

 

def fitness_evaluation(x, n_genes, n_alleles, scale, offset):
    x = binary2Decimal(x, n_genes, n_alleles, scale, offset)
    fitness = abs(0.25-np.sin(x))
    return fitness


def triangle_fitness(chromosome, n_genes, n_alleles, scale, offset):
    [a, b] = binary2Decimal(chromosome, n_genes, n_alleles, scale, offset)
    
    if a <= 0 or b <= 0:
        return 0

    c = np.sqrt(a**2 + b**2)
    perimeter = a + b + c
    print("perimeter: ",perimeter)
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    print("area", area)

    
    fitness = area / perimeter
    
    return fitness


def final_result(best_chromosome, best_fitness, n_genes, n_alleles, scale, offset ):
    best_chromosome = binary2Decimal(best_chromosome, n_genes, n_alleles, scale, offset)
    sine_value = np.sin(best_chromosome)
    print("#########################################333")
    print(f"Best fitness:{best_fitness}")
    print(f"best candidate solution: {best_chromosome}")
    print(f"sinus value obtainded: {sine_value}")
    print("#########################")
