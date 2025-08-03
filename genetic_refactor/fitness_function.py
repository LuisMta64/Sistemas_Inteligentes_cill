import numpy as np
import math
import pandas as pnd

def fitness_evaluation_triangle(decimal_values:list[float]):
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
    

def fitness_evaluation_parabola(decimal_values: list[float]) -> float:
    if len(decimal_values) != 3:
        raise ValueError("fitness_evaluation_parabola espera exactamente 3 genes.")
    a, b, c = decimal_values

    error_total = 0.0

    for x in range(-7, 8):
        target = 2 * x**2 + 3 * x + 1
        predicted = a * x**2 + b * x + c
        error_total += abs(target - predicted)

    return error_total


def fitness_evaluation_square(decimal_values: list[float]) -> float:
    if len(decimal_values) != 1:
        raise ValueError("Se espera solo 1 gen (el lado del cuadrado)")
    lado = decimal_values[0]
    if lado <= 0:
        return float('inf')

    area = lado ** 2
    perimeter = 4 * lado

    expected_area = 5.25 * 5.25 # ejemplo: 6x6
    expected_perimeter = 5.25 * 4 # 4*6

    diff_area = abs(expected_area - area)
    diff_perimeter = abs(expected_perimeter - perimeter)
    return diff_area + diff_perimeter

def fitness_evaluation_hexagon(decimal_values):
    if len(decimal_values) != 1:
        raise ValueError("Se espera solo 1 gen (el lado del hexágono regular)")

    lado = decimal_values[0]
    if lado <= 0:
        return float('inf')

    perimeter = 6 * lado
    area = (3 * math.sqrt(3) / 2) * lado**2

    expected_lado = 6
    expected_perimeter = 6 * expected_lado
    expected_area = (3 * math.sqrt(3) / 2) * expected_lado**2 

    diff_area = abs(expected_area - area)
    diff_perimeter = abs(expected_perimeter - perimeter)

    fitness = diff_area + diff_perimeter
    return fitness


def fitness_evaluation_quadratic_roots(decimal_values: list[float]) -> float:
    if len(decimal_values) != 3:
        raise ValueError("Se esperan exactamente 3 genes (a, b, c)")

    raices_deseadas = (2.0, 3.0)
    a, b, c = decimal_values
    #* SI LA A ES 0, ENTONCES NO ES CUADRATICA
    if a == 0:
        return 300

    #* APLICAMOS FORMULA GENERAL, SI ESTO DA NEGATIVO
    #* LA RAIZ SERA NEGATIVO (NUMERO IMAGINARIOP :p)
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return 300

    #* LA RAIZ DE LA FORMULA GENERAL
    sqrt_discriminant = math.sqrt(discriminant)

    #* LAS RAICES DEL +- DE LA FORMULA GENERAL
    r1 = (-b + sqrt_discriminant) / (2 * a)
    r2 = (-b - sqrt_discriminant) / (2 * a)

    #* ORDENAMOS PARA COMPARACION SIMETRICA ENTRE LAS OBTENIDAS Y ESPERADAS
    expected_r1, expected_r2 = sorted(raices_deseadas)
    calculated_r1, calculated_r2 = sorted([r1, r2])

    #* QUE TANTO NOS ALEJAMOS A LAS ESPERADAS (EL FITNESS)
    diff_r1 = abs(expected_r1 - calculated_r1)
    diff_r2 = abs(expected_r2 - calculated_r2)

    return diff_r1 + diff_r2



def fitness_diametro_circle(decimal_values):
    if len(decimal_values) != 1:
        raise ValueError("Se espera un solo valor: el diámetro del círculo")

    diametro = decimal_values[0]
    
    if diametro <= 0:
        return 99999
    diametro_esperado = 5.25
    radio_esperado = diametro_esperado / 2
    area_esperada = math.pi * radio_esperado**2
    perimetro_esperado = math.pi * diametro_esperado
    radio = diametro / 2
    area = math.pi * radio**2
    perimetro = math.pi * diametro
    diff_area = abs(area_esperada - area)
    diff_perimetro = abs(perimetro_esperado - perimetro)
    return diff_area + diff_perimetro

