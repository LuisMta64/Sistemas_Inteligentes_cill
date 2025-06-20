import population_generate as pg
import utils
import models as m

generations = 100
counter = 0
configuration = m.Configuration( n_gens=1, n_allels_per_gen=10 )
population = pg.generate_population( n_population=10, config=configuration )
pg.evaluate_fitness_population( population )
utils.sort_population_by_fitness( population )
while counter < generations:
    father1 = utils.select_person_by_roulette( population )
    father2 = utils.select_person_by_roulette( population )
    [son1, son2] = utils.get_crossover_sons( father1, father2, configuration )
    population.append( son1 )
    population.append( son2 )
    utils.sort_population_by_fitness( population )
    population = population[0:-2]
    counter +=1
    print(f"generation: {counter}, best_Fitness: {population[0].fitness}, worst_fitness: {population[-1].fitness}")

print(population[0].decimal_value)