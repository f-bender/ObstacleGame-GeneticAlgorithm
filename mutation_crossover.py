import numpy
from settings_constants import mutation_rate, variance

def get_new_population(old_pop):
    """ old_pop:    list of tuples: (list of numpy matrices) with their fitnesses as values

        returns:    list of list of numpy matrices
    """
    new_pop = []

    # print(f'old avg: {sum(x[1] for x in old_pop)/len(old_pop)}')
    
    for x in range(len(old_pop)//2):
        fitness_sum = sum(x[1] for x in old_pop)
        rand = numpy.random.uniform(high=fitness_sum)
        # print(f'sum={fitness_sum}, rand={rand}')
        running_sum = 0
        for idx, (matrices, fitness) in enumerate(old_pop):
            running_sum += fitness
            if running_sum > rand:
                # old_pop.remove((matrices, fitness))
                new_pop.append(old_pop.pop(idx)[0])
                break

    # print(f'new avg: {sum(x[1] for x in new_pop)/len(new_pop)}')

    new_pop2 = [mutated(matrices, mutation_rate, variance) for matrices in new_pop]

    new_pop = new_pop + new_pop2

    return new_pop

def mutated(matrices, mut_rate, var):
    mutated = [matrix.copy() for matrix in matrices]
    for matrix in mutated:
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                if numpy.random.uniform() < mut_rate:
                    matrix[x,y] = numpy.random.normal(matrix[x,y],var)  # maybe always keep the weights in the range (-1,1) ?
    return mutated

    # 0 is the mean of the normal distribution you are choosing from
    # 1 is the standard deviation of the normal distribution
    # 100 is the number of elements you get in array noise
        # noise = np.random.normal(0,1,100)


