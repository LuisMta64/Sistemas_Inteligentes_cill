import models as m
import numpy as np
import fitness_function
from typing import Optional
import matplotlib.pyplot as plt

evaluate_fitness = fitness_function.fitness_evaluation_parabola

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



def uniform_mutation( personToMutate: m.Person, probabilityToMutate: float = 0.1 ):
    chromosomeToMutate = personToMutate.chromosome
    mutated_chromosome = [ not (chromosomeToMutate[i]) if np.random.rand() <= probabilityToMutate else chromosomeToMutate[i] for i in range(len(chromosomeToMutate) ) ]
    personToMutate.set_chromosome( mutated_chromosome )

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

def graph_fitnesses(best_fitnesses, worst_fitnesses):
    plt.figure(figsize=(10, 6))
    generations = range(1, len(best_fitnesses) + 1)
    plt.plot(generations, best_fitnesses, label='Mejor fitness', color='green')
    plt.plot(generations, worst_fitnesses, label='Peor fitness', color='red')
    plt.title("Evolución de Fitness por Generación")
    plt.xlabel("Generación")
    plt.ylabel("Valor de Fitness")
    plt.grid(True, which='both', alpha=0.5)
    plt.legend()
    plt.xlim(1, len(best_fitnesses))
    if best_fitnesses and worst_fitnesses:
        y_min = min(min(best_fitnesses), min(worst_fitnesses))
        y_max = max(max(best_fitnesses), max(worst_fitnesses))
        plt.ylim(y_min * 0.9, y_max * 1.1)
    plt.show()




