import numpy as np
import random
from GA.evolution import *
from snake import Snake


class Population():

    def __init__(self,population):
        self.population = population
        self.size = len(population)
        self.generation = 0
        self.fitness_score_over_generations = np.zeros(shape=(1,1))
        self.best_indiviudal = None
        self.best_fitness = 0

    def selection(self, amount):
        sorted_pop = np.array(sorted(self.population, key=lambda x: x.fitness, reverse=True))
        return sorted_pop[:amount]

    def roulette_wheel_selection(self,selection_snakes, num_individuals):
        selection = []
        wheel = sum([snake.fitness for snake in selection_snakes])
        for _ in range(num_individuals):
            pick = np.random.uniform(0, wheel)
            current = 0
            for snake in selection_snakes:
                current += snake.fitness
                if current > pick:
                    selection.append(snake)
                    break
        return selection

    def reproduce(self, selection_of_snakes,prob_mutation):
        CANVAS = selection_of_snakes[0].canvas
        new_population = selection_of_snakes
        while len(new_population) < self.size:
            parent1,parent2 = self.roulette_wheel_selection(selection_of_snakes,2)
            num_layers = len(parent1.brain.layers)
            c1_params = {}
            c2_params = {}
            for l in range(num_layers-1):
                p1W = parent1.brain.parameters.get('W' + str(l))
                p1B = parent1.brain.parameters.get('B' + str(l))
                p2W = parent2.brain.parameters.get('W' + str(l))
                p2B = parent2.brain.parameters.get('B' + str(l))
                mutated_p1W = mutate(p1W,prob_mutation)
                mutated_p1B = mutate(p1B, prob_mutation)
                mutated_p2W = mutate(p2W, prob_mutation)
                mutated_p2B = mutate(p2B, prob_mutation)
                if np.random.uniform(0,1) < 0.5:
                    c1W, c2W = element_wise_crossover(mutated_p1W,mutated_p2W)
                    c1B, c2B = element_wise_crossover(mutated_p1B, mutated_p2B)
                else:
                    c1W, c2W = row_wise_crossover(mutated_p1W, mutated_p2W)
                    c1B, c2B = row_wise_crossover(mutated_p1B, mutated_p2B)
                c1_params['W' + str(l)] = c1W
                c1_params['B' + str(l)] = c1B
                c2_params['W' + str(l)] = c2W
                c2_params['B' + str(l)] = c2B
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['B' + str(l)], -1, 1, out=c1_params['B' + str(l)])
                np.clip(c2_params['B' + str(l)], -1, 1, out=c2_params['B' + str(l)])

            c1 = Snake(CANVAS)
            c2 = Snake(CANVAS)
            c1.brain.set_parameters(c1_params)
            c2.brain.set_parameters(c2_params)
            new_population = np.concatenate((new_population,[c1,c2]))
            new_population = new_population[:self.size]
            random.shuffle(new_population)
        return new_population

    def set_population(self,population):
        self.generation += 1
        self.population = population

    def set_best_individual(self,indiviudal,fitness):
        self.best_indiviudal = indiviudal
        self.best_fitness = fitness