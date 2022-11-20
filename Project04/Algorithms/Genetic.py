from typing import List, Dict

from Project04.NeuralNetwork import NeuralNetwork
from Selection import TournamentSelect
from Selection import Selection


class Genetic():
    def __init__(self, networks: List[NeuralNetwork], hyper_parameters: Dict[str, int], selection: Selection = TournamentSelect):
        self.population = networks

        # TODO: Handle hyper parameters
        self.hyper_parameters = hyper_parameters
        self.selection = selection(networks, self._evaluate_fitness)

    def _evaluate_fitness(self, chromosome: NeuralNetwork) -> float:
        return 0.0

    def train(self, num_generations: int, fitness_threshold: float) -> NeuralNetwork:
        # Compute the fitness for each chromosome in the population
        population_fitness = [self._evaluate_fitness(chromosome) for chromosome in self.population]

        for generation in range(num_generations):
            # Perform selection
            parents = self.selection.select()

            # Perform crossover
            children = self.crossover.cross(parents)

            # Mutate data
            children = self.mutation.mutate(children)

            # Evaluate fitness of children
            children_fitness = [self._evaluate_fitness(chromosome) for chromosome in children]

            # Update population

            # Terminate the training process if fitness threshold is satisfied
            if(self._evaluate_fitness() >= fitness_threshold):
                break
