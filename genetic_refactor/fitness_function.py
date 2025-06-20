import numpy as np

def fitness_evaluation(decimal_values):  # Recibe directamente el valor decimal
    fitness = abs(0.25-np.sin(decimal_values))
    return fitness 