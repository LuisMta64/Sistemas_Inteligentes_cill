import numpy as np
import import_ipynb
import sinusfitnesfunction as ff
from IPython.display import clear_output

class Genetic_Algorithm():
    def __init__(
            self,                   n_genes,                n_alleles, 
            scale,                  offset,                 size_population, 
            generations,            pm,                     min_fitness_value=0, 
            show_output = True,     tournamentSize=0,       parentsSize = 2, 
            randomMutation = True):
        
        #obligatory
        self.n_genes = n_genes
        self.n_alleles = n_alleles
        self.scale = scale
        self.offset = offset
        self.total_alleles = self.n_genes * self.n_alleles
        self.size_population = size_population
        self.chromosomes = []
        self.fitness = []
        self.generations = generations
        self.pm = pm

        ##optionals
        self.min_fitness_value = min_fitness_value
        self.show_output = show_output
        self.tournamentSize = tournamentSize
        self.parentsSize = parentsSize
        self.randomMutation = randomMutation

    ##opciones para iniciar una población
    def init_population(self):
        self.chromosomes = np.array([[1 if np.random.rand() < 0.5 else 0 for j in range(self.total_alleles)] for i in range(self.size_population)])

        self.fitness = np.array([ff.fitnessFunctionInference(self.chromosomes[i], self.n_genes, self.n_alleles, self.scale, self.offset) 
                    for i in range(self.size_population)])
    
    def random_selection(self):
        parents_index = np.random.randint(0, self.size_population, 2)
        return parents_index
    

    ##opciones para escoger padres en un offspring
    def tournament(self):
        mate = []
        for i in range(self.parentsSize):
            parentIndex = []
            a = np.random.randint(0, self.size_population)
            for j in range(1, self.tournamentSize):
                parentIndex.append(np.random.randint(0 , (self.size_population -1) ))
            mate.append( min(parentIndex) )
        return mate
    
    def tournament_profe(self):
        matting_pool = []
        for i in range (2):
            a = np.random.randint(0, self.size_population)
            for j in range(1, self.tournamentSize):
                a = np.minimum(a, np.random.randint(0, self.size_population))
            matting_pool.append(a)
        return matting_pool

    def routlete_wheel_selection(self):
        i = 0
        prob_population =  1 / (1 + (self.fitness) - self.min_fitness_value)
        prob_population_sum = np.sum(prob_population)
        prob_ind_i = prob_population[i] / prob_population_sum
        sum = prob_ind_i
        r = np.random.rand()
        while sum <= r:
            i += 1
            prob_ind_i = prob_population[i] / prob_population_sum 
            sum += prob_ind_i
        return i
    

    #offspring
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(0, self.total_alleles)
        offspring = []
        offspring.append(np.concatenate((parent1[0:crossover_point], parent2[crossover_point:])))
        offspring.append(np.concatenate((parent2[0:crossover_point], parent1[crossover_point:])))
        offspring = np.array(offspring)
        return offspring
    
    #opciones de mutaciones
    def uniform_mutation(self, chromosome):
        mutated_chromosome = [not (chromosome[i]) if np.random.rand() <= self.pm else chromosome[i] for i in range(self.total_alleles)]
        return np.array(mutated_chromosome)
    
    def inorder_mutation(self, chromosome):
        firstAllele = np.random.randint(0,self.n_alleles-1)
        secondAllele = np.random.randint(0,self.n_alleles-1)
        mutated_chromosome = chromosome.copy()
        if np.random.rand() <= self.pm:
            mutated_chromosome[firstAllele] = chromosome[secondAllele]
            mutated_chromosome[secondAllele] = chromosome[firstAllele]
        return np.array(mutated_chromosome)
    
    #ordenar población
    def sort_population(self):
        idx = np.argsort(self.fitness, axis=0).squeeze()
        self.chromosomes = self.chromosomes[idx]
        self.fitness = self.fitness[idx]

   
        
    
    
    #el mero mole
    def start(self):
        self.init_population()
        generation_counter = 0
        self.sort_population()
        array_best_fitness = [self.fitness[0][0]]
        array_worst_fitness = [self.fitness[-1][0]]
        while generation_counter <= self.generations:
            #Escogemos a los papases
            if self.tournamentSize > 0:
                 winnerParents = self.tournament()
                 id_parent1 = winnerParents[0]
                 id_parent2 = winnerParents[1]
            else:
                id_parent1 = self.routlete_wheel_selection()
                id_parent2 = self.routlete_wheel_selection()

            #Nace el neño
            offspring = self.crossover(self.chromosomes[id_parent1], self.chromosomes[id_parent2])

            #A lo mejor se deforma el niño
            if self.randomMutation:
                offspring_mutated = np.array([self.uniform_mutation(offspring[i]) for i in range(2)])
            else:
                offspring_mutated = np.array([self.inorder_mutation(offspring[i]) for i in range(2)])

            #vemos qué tan bueno salió el niño
            self.chromosomes = np.vstack((self.chromosomes, offspring_mutated))
            
            offspring_fitness = np.array(([ff.fitnessFunctionInference(offspring_mutated[i], self.n_genes, self.n_alleles, self.scale, self.offset) for i in range (2)]))
            self.fitness = np.vstack((self.fitness, offspring_fitness))
                
            #Ordenamos todo
            self.sort_population()
            self.chromosomes = self.chromosomes[0:self.size_population] #mataron mataron a un inocenteee
            self.fitness = self.fitness[0:self.size_population]

            #aumenta la generación
            generation_counter+=1

            #Pura estadística a partir de este punto
            best_fitness = self.fitness[0][0]
            worst_fitness = self.fitness[-1][0]
            array_best_fitness.append(best_fitness)
            array_worst_fitness.append(worst_fitness)
            #clear_output()
            if generation_counter % 100 == 0 and self.show_output:
                print(f"generation: {generation_counter}, best_Fitness: {best_fitness}, worst_fitness: {worst_fitness}")
            
        print("entré en el que no me sale")
        ff.Results_fitnessInference(self.chromosomes[0], array_worst_fitness, array_best_fitness, self.n_genes, self.n_alleles, self.scale, self.offset)
        


# Genetic_AlgorithmGrandote = Genetic_Algorithm(
#     n_genes=24, 
#     n_alleles=11, 
#     scale=20, 
#     offset=0, 
#     size_population=200,
#     generations=5000, 
#     pm=0.14, 
#     show_output=True, 
#     tournamentSize=20,
#     parentsSize = 2,
#     randomMutation = False
# )

Genetic_AlgorithmGrandote = Genetic_Algorithm(
    n_genes=24,
    n_alleles=9,
    scale=50,
    offset=0,
    size_population=100,
    generations=1000,
    pm=0.3,          
    show_output=True,
    tournamentSize=5,
    parentsSize=2,
    randomMutation=True,
)


# Genetic_AlgorithmGrandote = Genetic_Algorithm(
#     n_genes=24,
#     n_alleles=7,  # Bien
#     scale=20,
#     offset=0,
#     size_population=2,  # Reducir para mayor presión selectiva
#     generations=1,     # No necesitas 5000 si converge antes
#     # generations=1,     # No necesitas 5000 si converge antes
#     pm=0.5,             # Aumentar probabilidad de mutación
#     show_output=True,
#     tournamentSize=5,    # Reducir tamaño del torneo
#     parentsSize=2,
#     randomMutation=True, # Activado para mayor diversidad
# )
Genetic_AlgorithmGrandote.start()
