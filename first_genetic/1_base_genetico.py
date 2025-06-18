import numpy as np
import fitness_evaluation as fe

class Genetic_Algorithm():
    
    def __init__(self, population_size, n_alleles, n_genes, scale, offset, n_generaciones, min_fitness_value = 0):
        self.n_alleles = n_alleles
        self.n_genes = n_genes
        self.total_alleles = n_alleles*self.n_genes
        self.population_size = population_size
        self.scale = scale
        self.offset = offset
        self.n_generaciones = n_generaciones
        self.min_fitness_value = min_fitness_value
        self.chromosomes = []
        self.fitness = []

        self.decimal_value = []

    def generate_random_population(self):
        self.chromosomes = np.array([[1.0 if np.random.rand()>0.5 else 0 for i in range(self.total_alleles)]
                                        for i in range(self.population_size)])
        self.fitness = np.array([
            fe.fitness_evaluation(self.n_alleles, self.n_genes, self.scale, self.offset, self.chromosomes[i])[0]
                        for i in range(self.population_size)])
        self.decimal_value = np.array([
            fe.fitness_evaluation(self.n_alleles, self.n_genes, self.scale, self.offset, self.chromosomes[i])[1]
                        for i in range(self.population_size)])
    
    def sort_population(self):
        sorted_indices = np.argsort(self.fitness)[::-1]
        
        self.fitness = self.fitness[sorted_indices]
        self.chromosomes = self.chromosomes[sorted_indices]
        self.decimal_value = self.decimal_value[sorted_indices]

    def random_selection(self):
        parent_id = np.random.randint( 0, self.population_size )
        return parent_id
    def roulette_wheel_selection( self ):
        i = 0
        scale_fitness = 1/(1+ self.fitness - self.min_fitness_value )
        scale_fitness_sum = np.sum( scale_fitness )
        prob_i = scale_fitness[i] / scale_fitness_sum
        summ = prob_i
        r = np.random.rand()
        while summ<r:
            i+=1
            prob_i = scale_fitness[i] / scale_fitness_sum
            summ += prob_i
        return i
    def crossover(self, parent_1, parent_2):
        crossover_point = np.random.randint( 0, self.total_alleles )
        offspring = []
        offspring.append(np.concatenate( (parent_1[0:crossover_point], parent_2[crossover_point:]) ))
        offspring.append(np.concatenate( (parent_2[0:crossover_point], parent_1[crossover_point:]) ))
        return np.array( offspring )
    
    def start(self):
        self.generate_random_population()
        generation_counter = 0
        while generation_counter < self.n_generaciones:
            parents_id1 = self.roulette_wheel_selection()
            parents_id2 = self.roulette_wheel_selection()
            offspring = self.crossover(self.chromosomes[parents_id1], self.chromosomes[parents_id2])
            offspring_eval = [
                fe.fitness_evaluation(self.n_alleles, self.n_genes, self.scale, self.offset, child)
                for child in offspring
            ]
            
            fitness_values = np.array([val[0] for val in offspring_eval])
            decimal_values = np.array([val[1] for val in offspring_eval])
            self.chromosomes = np.vstack((self.chromosomes, offspring))
            self.fitness = np.append(self.fitness, fitness_values)
            self.decimal_value = np.append(self.decimal_value, decimal_values, axis=0)
            
            self.sort_population()
            self.chromosomes = self.chromosomes[:self.population_size]
            self.fitness = self.fitness[:self.population_size]
            self.decimal_value = self.decimal_value[:self.population_size]

            best_fitness = self.fitness[0]
            worst_fitness = self.fitness[-1]
            generation_counter += 1
            
            print(f'N generacion: {generation_counter} Best Fitness {best_fitness:.2f}, Worst Fitness {worst_fitness:.2f}')

genetic_algorithm1 = Genetic_Algorithm(
    population_size=500,
    n_alleles=5,
    n_genes=3,
    scale=10,
    offset=0,
    n_generaciones=1000
)
genetic_algorithm1.start()
# genetic_algorithm1.generate_random_population()
# print( genetic_algorithm1.chromosomes )
# print( genetic_algorithm1.fitness )
# genetic_algorithm1.sort_population()
# print( genetic_algorithm1.chromosomes )
# print( genetic_algorithm1.fitness )

# value_max = np.max(genetic_algorithm1.fitness)
# chromosoma_with_max_value = genetic_algorithm1.chromosomes[ np.argmax(genetic_algorithm1.fitness) ] 
# catetosPeores = genetic_algorithm1.decimal_value[ np.argmax(genetic_algorithm1.fitness) ] 

# value_min = np.min(genetic_algorithm1.fitness)
# chromosoma_with_min_value = genetic_algorithm1.chromosomes[ np.argmin(genetic_algorithm1.fitness) ]

# catetosMejores = genetic_algorithm1.decimal_value[ np.argmin(genetic_algorithm1.fitness) ] 

# print(f"Peor fitness: { value_max }, Catetos: { catetosPeores }, Chromosoma: {chromosoma_with_max_value}")
# print(f"Mejor fitness: {value_min}, Catetos: { catetosMejores }, Chromosoma: {chromosoma_with_min_value}")
