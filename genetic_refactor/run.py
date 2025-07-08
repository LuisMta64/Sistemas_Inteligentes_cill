import population_generate as pg
import utils
import models as m
import numpy as np


# PARA SENOS
# generations = 500
# counter = 0
# configuration = m.Configuration( n_gens=2, n_allels_per_gen=10, scale=100, offset=0 )
# population = pg.generate_population( n_population=100, config=configuration )

# PARA TRIANGULO
generations = 500
counter = 0
configuration = m.Configuration(n_gens=2, n_allels_per_gen=10, scale=10, offset=0)
population = pg.generate_population( n_population=100, config=configuration )


pg.evaluate_fitness_population( population )
utils.sort_population_by_fitness( population )

best_fitnesses = [  ]
worst_fitnesses = [  ]
while counter < generations:
    utils.sort_population_by_fitness( population )
    father1 = utils.select_person_by_roulette( population )
    father2 = utils.select_person_by_roulette( population )
    sons = utils.get_crossover_sons( father1, father2, configuration )
    [son1, son2] = sons

    utils.uniform_mutation(son1)
    utils.uniform_mutation(son2)

    population.append( son1 )
    population.append( son2 )
    utils.sort_population_by_fitness( population )
    population = population[0:-2]
    counter +=1
    best_fitnesses.append( population[0].fitness )
    worst_fitnesses.append( population[-1].fitness )
    
    if( counter % 100 == 0 ):
        print(f"generation: {counter}, best_Fitness: { population[0].fitness }, worst_fitness: {population[-1].fitness}")


print( f" Decimal: {population[0].decimal_value}, mejor fitness { population[0].fitness } " )
print( f" Decimal: {population[-1].decimal_value}, peor fitness { population[-1].fitness } " )
# print( f" Decimal: {population[0].decimal_value}, valuado en seno { np.sin(population[0].decimal_value) }, mejor fitness { population[0].fitness } " )
utils.graph_fitnesses( best_fitnesses, worst_fitnesses )
