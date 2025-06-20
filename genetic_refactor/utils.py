import models as m
import numpy as np
import fitness_function
from typing import Optional

def binary_to_decimal(chromosome, config: m.Configuration):
    list_values = []
    for i in range(config.n_gens):
        value = 0
        for j in range(config.n_allels_per_gen):
            bit_index = i * config.n_allels_per_gen + j
            value += chromosome[bit_index] * 2**(config.n_allels_per_gen - 1 - j)
        decimal = value / (2**config.n_allels_per_gen - 1) * config.scale + config.offset
        list_values.append(decimal)
    return np.array(list_values)

def sort_population_by_fitness( persons: list[m.Person] ):
    persons.sort(key=lambda person: person.fitness)

def select_person_by_roulette( persons: list[m.Person], min_fitness_value = 0 ):
    fitness = np.array([ i.fitness for i in persons])
    indice_seleccionado = 0
    prob_population =  1 / (1 + (fitness) - min_fitness_value)
    prob_population_sum = np.sum(prob_population)
    prob_ind_i = prob_population[indice_seleccionado] / prob_population_sum
    sum1 = prob_ind_i
    r = np.random.rand()
    while sum1 <= r:
        indice_seleccionado += 1
        prob_ind_i = prob_population[indice_seleccionado] / prob_population_sum 
        sum1 += prob_ind_i
    return persons[indice_seleccionado]

def get_crossover_sons(father1: m.Person, father2: m.Person, config: m.Configuration):
    crossover_point = np.random.randint(0, config.total_allels)
    offspring = [
        np.concatenate((father1.chromosome[:crossover_point], father2.chromosome[crossover_point:])),
        np.concatenate((father2.chromosome[:crossover_point], father1.chromosome[crossover_point:]))
    ]
    sons = []
    for chrom in offspring:
        decimal = binary_to_decimal(chrom.tolist(), config)
        fitness = fitness_function.fitness_evaluation(decimal)
        sons.append(m.Person(
            chromosome=chrom.tolist(),
            decimal_value=decimal,
            fitness=fitness
        ))
    return sons

def print_chromosomes( chromosomes: list[m.Person] = None):
    for individuo in chromosomes:
        print( individuo.chromosome )

def print_fitness( chromosomes: list[m.Person] = None):
    for individuo in chromosomes:
        print( individuo.fitness )

def print_decimal_values( chromosomes: list[m.Person] = None):
    for individuo in chromosomes:
        print( individuo.decimal_value )

def print_population( chromosomes: list[m.Person] = None ):
    for individuo in chromosomes:
        print( f" fitness: { individuo.fitness }, chromosome: { individuo.chromosome }, decimal: { individuo.decimal_value }, { np.sin(individuo.decimal_value) }" )

