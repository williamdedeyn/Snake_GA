import numpy as np


def row_wise_crossover(chromosome1: np.ndarray, chromosome2: np.ndarray) -> (np.ndarray, np.ndarray):
    if chromosome1.shape != chromosome2.shape:
        raise Exception('Make sure the 2 parents have the same dimensions')
    array1 = chromosome1.flatten()
    array2 = chromosome2.flatten()
    swap_point = np.random.randint(1, len(array1) - 2)

    array1[:swap_point] = array2[:swap_point]
    array2[swap_point:] = array1[swap_point:]

    offspring1 = array1.reshape(chromosome1.shape)
    offspring2 = array2.reshape(chromosome1.shape)
    return offspring1, offspring2


def element_wise_crossover(chromosome1: np.ndarray, chromosome2: np.ndarray) -> (np.ndarray, np.ndarray):
    offspring1 = chromosome1.copy()
    offspring2 = chromosome2.copy()

    mask = np.random.uniform(0, 1, size=offspring1.shape)
    offspring1[mask > 0.5] = chromosome2[mask > 0.5]
    offspring2[mask > 0.5] = chromosome1[mask > 0.5]

    return offspring1, offspring2


def mutate(chromosome: np.ndarray,prob:float) -> np.ndarray:
    mutate_array = np.random.random(chromosome.shape) < prob
    mutations = np.random.uniform(-1, 1, size=chromosome.shape)
    chromosome[mutate_array] = mutations[mutate_array]

    return chromosome
