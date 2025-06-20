class Person: 
    def __init__(self, chromosome: list, decimal_value , fitness  = 0, ):
        self.chromosome = chromosome
        self.fitness = fitness
        self.decimal_value = decimal_value

    def set_fitness(self, fitness):
        self.fitness = fitness
    def set_decimal_value(self, decimal_value):
        self.decimal_value = decimal_value

class Configuration:
    def __init__(self, n_gens: int, n_allels_per_gen: int, scale: int = 100, offset: int = 0):
        self.n_gens = n_gens
        self.scale = scale
        self.offset = offset
        self.n_allels_per_gen = n_allels_per_gen
        self.total_allels = n_allels_per_gen * n_gens
