import numpy as np
import sinusfitnesfunction as ff

class Genetic_Algorithm():
    def __init__(self, n_genes, n_alleles, scale, offset, size_population, generations, min_fitness_value=0):
        self.n_genes = n_genes
        self.n_alleles = n_alleles
        self.scale = scale
        self.offset = offset
        self.total_alleles = self.n_genes * self.n_alleles
        self.size_population = size_population
        self.chromosomes = []
        self.fitness = []
        self.min_fitness_value = min_fitness_value
        self.generations = generations
    def init_population(self):
        self.chromosomes = np.array([[1 if np.random.rand() < 0.5 else 0 for j in range(self.total_alleles)] for i in range(self.size_population)])

        self.fitness = np.array([ff.fitness_evaluation(self.chromosomes[i], self.n_genes, self.n_alleles, self.scale, self.offset) 
                        for i in range(self.size_population)])
    
    def triangle_fitness(self):
        self.trianglefitness = [ff.triangle_fitness(self.chromosomes[i], self.n_genes, self.n_alleles, self.scale, self.offset) 
                               for i in range(self.size_population)]
    def random_selection(self):
        parents_index = np.random.randint(0, self.size_population, 2)
        return parents_index
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
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(0, self.total_alleles)
        offspring = []
        offspring.append(np.concatenate((parent1[0:crossover_point], parent2[crossover_point:])))
        offspring.append(np.concatenate((parent2[0:crossover_point], parent1[crossover_point:])))
        offspring = np.array(offspring)
        return offspring
    def sort_population(self):
        new_Order = list(range(self.fitness.size))
        
        for i in range(len(new_Order) - 1):
            for j in range(len(new_Order) - 1 - i):
                if self.fitness[new_Order[j]] > self.fitness[new_Order[j + 1]]:
                    temp = new_Order[j]
                    new_Order[j] = new_Order[j + 1]
                    new_Order[j + 1] = temp

        self.fitness = self.fitness[new_Order]
        self.chromosomes = self.chromosomes[new_Order]

    def start(self):
        self.init_population()
        generation_counter = 0
        self.sort_population()
        while generation_counter <= self.generations:
            id_parent1 = self.routlete_wheel_selection()
            id_parent2 = self.routlete_wheel_selection()
            offspring = self.crossover(self.chromosomes[id_parent1], self.chromosomes[id_parent2])
            self.chromosomes = np.vstack((self.chromosomes, offspring))
            offspring_fitness = ([ff.fitness_evaluation(offspring[i], self.n_genes, self.n_alleles, self.scale, self.offset) for i in range (2)])
            self.fitness = np.vstack((self.fitness, offspring_fitness))
            self.sort_population()
            self.chromosomes = self.chromosomes[0:self.size_population] #mataron mataron a un inocenteee
            self.fitness = self.fitness[0:self.size_population]
            generation_counter+=1
            best_fitness = self.fitness[0][0]
            worst_fitness = self.fitness[-1][0]
            print(f"generation: {generation_counter}, best_Fitness: {best_fitness}, worst_fitness: {worst_fitness}")
        ff.final_result(self.chromosomes[0], best_fitness, self.n_genes, self.n_alleles, self.scale, self.offset)
            
Genetic_Algorithm1 = Genetic_Algorithm(1, 10, 100, 0, 100, 100)
Genetic_Algorithm1.start()