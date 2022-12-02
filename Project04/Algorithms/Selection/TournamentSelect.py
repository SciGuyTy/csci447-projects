import random
from typing import List, Callable

from Project04.Algorithms.Selection.Selection import Selection


class TournamentSelect(Selection):
    def __init__(self, fitness_function: Callable):
        super().__init__(fitness_function)

    def select(self, population: List[float], k: int = 3) -> List[float]:
        """
        Perform tournament selection on a population of chromosomes

        Parameters
        ----------
        population: List[float]
            The target population to select from

        k: int
            The number of chromosomes to participate in the tournament (defaults to 5)

        Returns
        -------
        The champion chromosome from the tournament
        """

        # Select random chromosomes from population (without replacement)
        contestants = random.sample(population, k)

        # Compute and compare fitness
        contestant_fitness = [self.fitness_function(contestant) for contestant in contestants]

        # Find index of contestant with best fitness
        best_fitness = min(contestant_fitness)
        index_of_selection = contestant_fitness.index(best_fitness)

        # Return the best contestant
        return contestants[index_of_selection]
