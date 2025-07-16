import population_generate as pg
import utils
import models as m
import numpy as np
import math 

generations = 4000
# generations = 100000
counter = 0
#* CUADRADO
configuration = m.Configuration(
    n_gens=1,
    n_allels_per_gen=20,
    scale=20,
    offset=0
)
#* ECUACION
# configuration = m.Configuration(
#     n_gens=3,
#     n_allels_per_gen=20,
#     scale=20,
#     offset=-10
# )
population = pg.generate_population( n_population=100, config=configuration )


pg.evaluate_fitness_population( population )
utils.sort_population_by_fitness( population )

best_fitnesses = [  ]
worst_fitnesses = [  ]
while counter < generations:
    utils.sort_population_by_fitness( population )
    father1 = utils.select_person_by_roulette( population )
    father2 = utils.select_person_by_roulette( population )
    sons = pg.get_crossover_sons( father1, father2, configuration )
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

# print(f"-----> ESPERAMOS RAICES {(2.0, 3.0)}")
# print(f"-----> MEJOR FITNESS")
# print( f"Decimales (A,B,C): {population[0].decimal_value}, fitness { population[0].fitness } " )
# sqrt_discriminant = math.sqrt((population[0].decimal_value[1])**2 - 4 * population[0].decimal_value[0] * population[0].decimal_value[2])
# r1 = (-population[0].decimal_value[1] + sqrt_discriminant) / (2 * population[0].decimal_value[0])
# r2 = (-population[0].decimal_value[1] - sqrt_discriminant) / (2 *  population[0].decimal_value[0])
# print( f"Raices obtenidas {r1} y {r2} " )
# print()
# print(f"-----> PEOR FITNESS")
# print( f"DECIMAL : {population[-1].decimal_value}, peor fitness { population[-1].fitness } " )
print(f"EXPECTED: AREA { math.pi * 5.25 **2} PERIMETRO : {5.25 * math.pi }")
print(f"BEST FITNESS")
print( f"(diametro): {population[0].decimal_value}, fitness { population[0].fitness } " )
print( f"AREA { math.pi * population[0].decimal_value **2} PERIMETRO : {population[0].decimal_value}" )
print()
print(f"WORST FITNESS")
print( f"DECIMAL : {population[-1].decimal_value}, peor fitness { population[-1].fitness } " )
utils.graph_fitnesses( best_fitnesses, worst_fitnesses )
