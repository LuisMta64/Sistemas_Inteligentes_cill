import numpy as np
import fitness_evaluation as fe
import matplotlib.pyplot as plt

class Genetic_Algorithm():
    
    def __init__(self, population_size, n_alleles, n_genes, scale, offset, n_generaciones, min_fitness_value=0):
        self.n_alleles = n_alleles
        self.n_genes = n_genes
        self.total_alleles = n_alleles * n_genes
        self.population_size = population_size
        self.scale = scale
        self.offset = offset
        self.n_generaciones = n_generaciones
        self.min_fitness_value = min_fitness_value
        self.chromosomes = []
        self.fitness = []
        self.decimal_value = []
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def generate_random_population(self):
        self.chromosomes = np.random.randint(0, 2, (self.population_size, self.total_alleles))
        self.fitness = np.array([fe.fitness_evaluation(self.n_alleles, self.n_genes, self.scale, 
                                                      self.offset, self.chromosomes[i])[0] 
                               for i in range(self.population_size)])
        self.decimal_value = np.array([fe.fitness_evaluation(self.n_alleles, self.n_genes, self.scale, 
                                                           self.offset, self.chromosomes[i])[1] 
                                     for i in range(self.population_size)])
    
    def sort_population(self):
        idx = np.argsort(self.fitness)[::-1]  # Orden descendente
        self.fitness = self.fitness[idx]
        self.chromosomes = self.chromosomes[idx]
        self.decimal_value = self.decimal_value[idx]

    def roulette_wheel_selection(self):
        # Normalización para asegurar valores positivos
        normalized_fitness = self.fitness - np.min(self.fitness) + 1e-10
        total_fitness = np.sum(normalized_fitness)
        prob = normalized_fitness / total_fitness
        return np.random.choice(len(self.fitness), p=prob)

    def crossover(self, parent_1, parent_2):
        crossover_point = np.random.randint(1, self.total_alleles-1)
        offspring_1 = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))
        offspring_2 = np.concatenate((parent_2[:crossover_point], parent_1[crossover_point:]))
        return np.array([offspring_1, offspring_2])
    
    def mutate(self, chromosome, mutation_rate=0.01):
        for i in range(len(chromosome)):
            if np.random.rand() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Flip the bit
        return chromosome

    def plot_evolution(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='Best Fitness')
        plt.plot(self.avg_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness')
        plt.legend()
        plt.grid()
        plt.show()

    def start(self):
        self.generate_random_population()
        self.sort_population()
        
        for generation in range(self.n_generaciones):
            new_population = []
            new_fitness = []
            new_decimal = []
            
            # Elitismo: mantener al mejor individuo
            new_population.append(self.chromosomes[0])
            new_fitness.append(self.fitness[0])
            new_decimal.append(self.decimal_value[0])
            
            while len(new_population) < self.population_size:
                # Selección
                parent1_id = self.roulette_wheel_selection()
                parent2_id = self.roulette_wheel_selection()
                
                # Crossover
                offspring = self.crossover(self.chromosomes[parent1_id], 
                                          self.chromosomes[parent2_id])
                
                # Mutación
                offspring[0] = self.mutate(offspring[0])
                offspring[1] = self.mutate(offspring[1])
                
                # Evaluación
                for child in offspring:
                    fit, dec = fe.fitness_evaluation(self.n_alleles, self.n_genes, 
                                                   self.scale, self.offset, child)
                    new_population.append(child)
                    new_fitness.append(fit)
                    new_decimal.append(dec)
            
            # Actualizar población
            self.chromosomes = np.array(new_population[:self.population_size])
            self.fitness = np.array(new_fitness[:self.population_size])
            self.decimal_value = np.array(new_decimal[:self.population_size])
            
            self.sort_population()
            
            # Registrar estadísticas
            self.best_fitness_history.append(self.fitness[0])
            self.avg_fitness_history.append(np.mean(self.fitness))
            
            # Mostrar progreso
            if generation % 10 == 0:
                # print(f'Gen {generation:4d}: Best {self.fitness[0]:.6f}, '
                #       f'Avg {np.mean(self.fitness):.6f}, '
                #       f'Worst {self.fitness[-1]:.6f}')
                print( self.fitness )
        
        # Resultados finales
        print("\n=== Final Results ===")
        print(f"Best Fitness: {self.fitness[0]:.6f}")
        print(f"Best Solution: {self.decimal_value[0]}")
        print(f"Best Chromosome: {self.chromosomes[0]}")
        
        # Graficar evolución
        self.plot_evolution()
        
genetic_algorithm1 = Genetic_Algorithm(
    population_size=5,
    n_alleles=2,
    n_genes=2,
    scale=10,
    offset=0,
    n_generaciones=1000
)
genetic_algorithm1.start()