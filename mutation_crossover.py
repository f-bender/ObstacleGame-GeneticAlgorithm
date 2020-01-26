import numpy
from settings_constants import mutation_rate, variance

def get_new_population(old_pop):
    """ old_pop:    list of tuples: (list of numpy matrices) with their fitnesses as values

        returns:    list of list of numpy matrices
    """
    new_pop = []

    max_fitness = 0
    for _ in range(len(old_pop)//2):
        fitness_sum = sum(x[1] for x in old_pop)
        rand = numpy.random.uniform(high=fitness_sum)
        running_sum = 0
        for idx, (_, fitness) in enumerate(old_pop):
            running_sum += fitness
            if running_sum > rand or idx == len(old_pop)-1: # failsave: spätestens das lezte element wird auf jeden Fall genommen
                if fitness > max_fitness:
                    max_fitness = fitness
                    new_pop.insert(0,old_pop.pop(idx)[0])
                else:
                    new_pop.append(old_pop.pop(idx)[0])
                break

    new_pop_mutated = [mutateMatrices(matrices, mutation_rate, variance) for matrices in new_pop]
    new_pop = new_pop + new_pop_mutated

    return new_pop
    

def mutateMatrices(matrices, mut_rate, var):
    mutatedMatrices = [matrix.copy() for matrix in matrices]
    for matrix in mutatedMatrices:
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                if numpy.random.uniform() < mut_rate:
                    matrix[x,y] = numpy.random.normal(matrix[x,y],var)  # maybe always keep the weights in the range (-1,1) ?
    return mutatedMatrices