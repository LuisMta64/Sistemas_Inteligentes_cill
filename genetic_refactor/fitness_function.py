import numpy as np
import math
import pandas as pnd

def fitness_evaluation(decimal_values:list[float]):
    if len(decimal_values) != 2:
        raise ValueError("fitness_evaluation espera exactamente 2 genes.")
    a = decimal_values[0]
    b = decimal_values[1]
    if a <= 0 or b <= 0:
        return float('inf')
    c = math.sqrt(a**2 + b**2)
    area = (a * b) / 2
    perimeter = a + b + c
    expected_area = 12.5
    expected_perimeter = 5 + 5 + math.sqrt(5**2 + 5**2)
    diff_area = abs(expected_area - area)
    diff_perimeter = abs(expected_perimeter - perimeter)
    fitness = diff_area + diff_perimeter
    return fitness


def fitness_evaluation_diabetes( decimal_values: list[float] ):
    pnd
    return


# def fitness_evaluation(decimal_values):
#     fitness = abs(0.25-np.sin(decimal_values))
#     return fitness 