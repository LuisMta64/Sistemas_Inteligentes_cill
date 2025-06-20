import numpy as np
import utils
import models as m
import fitness_function 

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

def evaluate_population_decimal( persons: list[m.Person], config: m.Configuration ):
    for person in persons:
        person.set_decimal_value( utils.binary_to_decimal( person.chromosome, config ) )

def evaluate_fitness_population( persons: list[m.Person] ):
    for person in persons:
        person.set_fitness( fitness_function.fitness_evaluation( person.decimal_value ) )