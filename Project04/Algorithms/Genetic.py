import math
from typing import List, Dict, Callable

import numpy as np

from Project04.Algorithms.Crossover.UniformCrossover import UniformCrossover
from Project04.Algorithms.Mutation.UniformMutation import UniformMutation
from Project04.NeuralNetwork import NeuralNetwork
from Project04.Utilities.Utilities import Utilities
from Selection.TournamentSelect import TournamentSelect


class Genetic():
    def __init__(self, initial_networks: List[NeuralNetwork], hyper_parameters: Dict[str, object],
                 evaluation_method: Callable):
        # Initial population of networks to be used for training
        self.population = initial_networks

        # Method for evaluating the fitness of a chromosome
        self.evaluation_method = evaluation_method

        # Hyper-parameters
        self.hyper_parameters = hyper_parameters
        self.probability_of_cross = self.hyper_parameters['probability_of_cross']
        self.probability_of_mutation = self.hyper_parameters['probability_of_mutation']
        self.mutation_range = self.hyper_parameters['mutation_range']
        self.selection = self.hyper_parameters['selection'](evaluation_method)
        self.crossover = self.hyper_parameters['crossover'](self.probability_of_cross)
        self.mutation = self.hyper_parameters['mutation'](self.mutation_range, self.probability_of_mutation)
        self.num_replaced_parents = self.hyper_parameters['num_replaced_parents']
        self.tournament_size = self.hyper_parameters['tournament_size']

    def train(self, num_generations: int) -> tuple[NeuralNetwork, float]:
        # Breed and mutate the population for the given number of generations
        for generation in range(num_generations):
            # List to store children chromosomes
            generation_children = []

            # Replace parents with children
            for _ in range(math.floor(self.num_replaced_parents / 2)):
                parents = []

                # Perform selection to get two parent chromosomes and remove them from the current population
                for _ in range(2):
                    # Select a parent
                    parent = self.selection.select(self.population)

                    # Remove the parent from the population (ran into weird issues with np.array and python list types)
                    parent_index = [idx for idx, el in enumerate(self.population) if np.array_equal(el, parent)]
                    del self.population[parent_index[0]]

                    # Store the parent
                    parents.append(parent)

                # Perform crossover to generate children chromosomes
                children = self.crossover.cross(parents[0], parents[1])

                # Mutate children chromosomes and store them in the children array
                generation_children += (self.mutation.mutate(children))

            # Update population by inserting new children into population
            self.population += generation_children

        # Compute the fitness for each chromosome in the final population
        population_fitness = [self.evaluation_method(chromosome) for chromosome in self.population]

        best_fitness = min(population_fitness)
        best_fitness_index = population_fitness.index(best_fitness)

        # Return the best network and its fitness
        return self.population[best_fitness_index], best_fitness


if __name__ == "__main__":
    networks = []

    for i in range(10):
        nn = NeuralNetwork([3, 3], lambda x: x, False, (0, 1))
        serialized_nn = Utilities.serialize_network(nn)
        networks.append(serialized_nn)

    ga = Genetic(networks, {'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation,
                            'num_replaced_parents': 8, 'tournament_size': 3, 'probability_of_cross': 0.8, 'probability_of_mutation': 0.15, 'mutation_range': (0, 1)}, evaluation_method=lambda x: 0.0)
    print(ga.train(10))
