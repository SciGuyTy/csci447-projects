from typing import List, Dict, Callable

from Project04.Algorithms.Crossover.UniformCrossover import UniformCrossover
from Project04.Algorithms.Mutation.UniformMutation import UniformMutation
from Project04.NeuralNetwork import NeuralNetwork
from Project04.Utilities.Utilities import Utilities
from Selection.TournamentSelect import TournamentSelect
from Selection.Selection import Selection


class Genetic():
    def __init__(self, networks: List[NeuralNetwork], hyper_parameters: Dict[str, int]):
        self.population = networks

        # TODO: Handle hyper parameters
        self.hyper_parameters = hyper_parameters
        self.selection = self.hyper_parameters['selection'](networks, self._evaluate_fitness)
        self.crossover = self.hyper_parameters['crossover']()
        self.mutation = self.hyper_parameters['mutation']((0, 1))

    def _evaluate_fitness(self, chromosome: NeuralNetwork) -> float:
        # TODO: How to evaluate fitness?
        return 0.0

    def train(self, num_generations: int, fitness_threshold: float) -> NeuralNetwork:
        # Compute the fitness for each chromosome in the population
        population_fitness = [self._evaluate_fitness(chromosome) for chromosome in self.population]

        # Run for each sample in the training data
        for generation in range(num_generations):
            # Perform selection to get two parent chromosomes
            parents = [self.selection.select() for _ in range(2)]

            print(parents)

            # Perform crossover to generate children chromosomes
            children = self.crossover.cross(parents[0], parents[1])

            print(children)

            # Mutate children chromosomes
            children = self.mutation.mutate(children)

            print(children)

            # Evaluate fitness of children
            # children_fitness = [self._evaluate_fitness(chromosome) for chromosome in children]

            # Update population
            # Replace all parents in the population?

            # Terminate the training process if fitness threshold is satisfied
            # if(self._evaluate_fitness() >= fitness_threshold):
            #     break


if __name__ == "__main__":
    networks = []

    for i in range(10):
        nn = NeuralNetwork([3, 3], lambda x: x, False, (0, 1))
        serialized_nn = Utilities.serialize_network(nn)
        networks.append(serialized_nn)

    ga = Genetic(networks, {'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation})
    ga.train(1, 0.0)
