import random
from typing import List

from Project04.Algorithms.Mutation.Mutation import Mutation


class UniformMutation(Mutation):
    def __init__(self, mutation_range: tuple[float, float], probability_of_cross: float = 0.15):
        super().__init__(mutation_range, probability_of_cross)

    def mutate(self, chromosomes: List[List[float]]) -> List[List[float]]:
        # Create a copy of the chromosome objects for data consistency
        chromosomes = chromosomes.copy()

        for chromosome in chromosomes:
            for index in range(len(chromosome)):
                # Only perform a mutation based on the provided probability of a mutation occurring
                if random.random() > (1 - self.probability_of_mutation):
                    # Swap the two values
                    chromosome[index] = random.uniform(self.mutation_range[0], self.mutation_range[1])

        return chromosomes
