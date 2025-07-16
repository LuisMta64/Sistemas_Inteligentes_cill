import numpy as np
import utils
import models as m
import fitness_function 

evaluate_fitness = fitness_function.fitness_diametro_circle

def generate_population( n_population: int, config: m.Configuration ):
    total_alleles = config.n_gens * config.n_allels_per_gen
    persons = []
    for i in range(n_population):
        chromosome = []
        for j in range(total_alleles):
            if np.random.rand() < 0.5:
                chromosome.append(1)
            else:
                chromosome.append(0)
        
        person_decimal_value = utils.binary_to_decimal(
            chromosome=chromosome,
            config=config
        )
        individuo = m.Person( chromosome=chromosome , decimal_value=person_decimal_value )
        persons.append( individuo )
    return persons

def get_crossover_sons(father1: m.Person, father2: m.Person, config: m.Configuration):
    crossover_point = np.random.randint(0, config.total_allels)
    offspring = [
        np.concatenate((father1.chromosome[:crossover_point], father2.chromosome[crossover_point:])),
        np.concatenate((father2.chromosome[:crossover_point], father1.chromosome[crossover_point:]))
    ]
    sons = []
    for chrom in offspring:
        decimal = utils.binary_to_decimal(chrom.tolist(), config)
        fitness = evaluate_fitness(decimal)
        sons.append(m.Person(
            chromosome=chrom.tolist(),
            decimal_value=decimal,
            fitness=fitness
        ))
    return sons

def evaluate_population_decimal( persons: list[m.Person], config: m.Configuration ):
    for person in persons:
        person.set_decimal_value( utils.binary_to_decimal( person.chromosome, config ) )

def evaluate_fitness_population(persons: list[m.Person]):
    for person in persons:
        person.set_fitness(evaluate_fitness(person.decimal_value))