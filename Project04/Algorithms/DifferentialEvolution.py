import math
import random
from typing import List, Dict, Callable

import numpy as np

from Project04.Algorithms.Crossover.BinomialCrossover import BinomialCrossover
from Project04.NeuralNetwork import NeuralNetwork
from Project04.Utilities.Utilities import Utilities


class Genetic():
    def __init__(self, initial_networks: List[NeuralNetwork], hyper_parameters: Dict[str, object],
                 evaluation_method: Callable):
        # Initial population of networks to be used for training
        self.population = initial_networks

        # Method for evaluating the fitness of a chromosome
        self.evaluation_method = evaluation_method

        # Hyper-parameters
        self.hyper_parameters = hyper_parameters
        self.num_replaced_parents = self.hyper_parameters['num_replaced_parents']
        self.mutation_scale_factor = self.hyper_parameters['mutation_scale_factor']
        self.crossover_rate = self.hyper_parameters['crossover_rate']

        # Crossover method
        self.crossover = self.hyper_parameters['crossover'](self.crossover_rate)


    def train(self, num_generations: int) -> tuple[NeuralNetwork, float]:
        """
        Train a neural network using differential evolution

        Parameters
        ----------
        num_generations: int
            The number of generations to run the training algorithm

        Returns
        -------
        A tuple containing the best weight configuration (serialized) and the associated fitness
        """
        # Keep track of the best chromosome
        best_chromosome = ([], math.inf)

        # Breed and mutate the population for the given number of generations
        for generation in range(num_generations):
            for index, chromosome in enumerate(self.population):
                # Remove target chromosome from population
                del self.population[index]

                # Select three chromosomes without replacement from population
                a, b, c = random.sample(self.population, 3)

                # Perform differential mutation to produce trial chromosome
                trial_chromosome = a + (self.mutation_scale_factor * (np.subtract(b, c)))

                # Perform crossover
                trial_chromosome = self.crossover.cross(chromosome, trial_chromosome)

                # Compare trial vector and target vector using evaluation method
                target_fitness = self.evaluation_method(chromosome)
                trial_fitness = self.evaluation_method(trial_chromosome)

                if trial_fitness < target_fitness:
                    # Replace target chromosome with trial chromosome in population
                    self.population.insert(index, trial_chromosome)

                    # Keep track of the best chromosome
                    if trial_fitness < best_chromosome[1]:
                        best_chromosome = trial_chromosome, trial_fitness
                else:
                    # Reinsert the target chromosome in population
                    self.population.insert(index, chromosome)

                    # Keep track of the best chromosome
                    if target_fitness < best_chromosome[1]:
                        best_chromosome = chromosome, target_fitness

        # Return the best network and its fitness
        return best_chromosome


if __name__ == "__main__":
    networks = []

    for i in range(10):
        nn = NeuralNetwork([3, 3], lambda x: x, False, (0, 1))
        serialized_nn = Utilities.serialize_network(nn)
        networks.append(serialized_nn)

    ga = Genetic(networks, {'crossover': BinomialCrossover,  'num_replaced_parents': 8, 'mutation_scale_factor': 1, 'crossover_rate': 0.5}, evaluation_method=lambda x: 0.0)
    print(ga.train(10))
