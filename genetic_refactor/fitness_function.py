import numpy as np
def fitness_evaluation( decimal_value: float ):
    fitness = abs(0.25-np.sin(decimal_value))
    return fitness 