import math
import random
from typing import List, Dict, Callable

import numpy as np
from Project04.NeuralNetwork import NeuralNetwork
from Project04.Utilities.Utilities import Utilities

class Particle():

    def __init__(self, network: NeuralNetwork, evaluation_method: Callable, inertia: float, c1: float, c2: float):
        # Save evaluation method
        self.evaluate_self = evaluation_method

        self.original_network = network

        # Set the position as the serialized network
        self.position = Utilities.serialize_network(network)

        # Initialize the velocity to a zero vector
        self.velocity = 0 * self.position

        # Keep track of the best position (initialize to starting position)
        self.pbest_fitness = math.inf
        self.pbest_position = self.position

        # Keep track of hyperparameters
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

    def evaluate(self):
        return self.evaluate_self(Utilities.deserialize_network(self.original_network, self.position))

    def update(self, global_best_position):
        r1 = random.random()
        r2 = random.random()

        # Update velocity
        self.velocity = self.inertia * self.velocity + \
                        self.c1 * r1 * (global_best_position - self.position) + \
                        self.c2 * r2 * (self.pbest_position - self.position)

        # Update position
        self.position += self.velocity

        current_fitness = self.evaluate()
        if current_fitness < self.pbest_fitness:
            self.pbest_fitness = current_fitness
            self.pbest_position = self.position

        return current_fitness

class PSO():
    def __init__(self, networks: List[NeuralNetwork], hyper_parameters: Dict[str, float], evaluation_method: Callable):
        self.population = networks
        self.evaluation_method = evaluation_method
        self.hyper_parameters = hyper_parameters
        self.inertia = self.hyper_parameters['inertia']
        self.c1 = self.hyper_parameters['c1']
        self.c2 = self.hyper_parameters['c2']
        self.particles = [Particle(network, self.evaluation_method, self.inertia, self.c1, self.c2) for network in self.population]

    def train(self, num_generations: int) -> NeuralNetwork:
        # Compute the fitness for each chromosome in the population
        population_fitness = np.array([particle.evaluate() for particle in self.particles])

        # Keep track of the global best fitness
        gbest_index = np.argmin(population_fitness)
        gbest_position = self.particles[gbest_index].position
        gbest_fitness = population_fitness[gbest_index]

        gbest_position_new = gbest_position
        # Run for each generation
        for generation in range(num_generations):
            print("Starting generation: ", generation)

            # Loop through the particles
            for i, particle in enumerate(self.particles):
                # Update the particle and get its fitness
                fitness = particle.update(gbest_position)

                # If the particle has a better fitness than the global best fitness
                if fitness < gbest_fitness:
                    # Save the new best fitness and position
                    gbest_index = i
                    gbest_fitness = fitness
                    gbest_position_new = particle.position

            # Delay updating the global best position until all particles are updated in the generation
            gbest_position = gbest_position_new
            print("Finished generation: ", generation)
        return Utilities.deserialize_network(self.population[gbest_index], gbest_position), gbest_fitness



