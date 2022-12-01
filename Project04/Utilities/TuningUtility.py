import multiprocessing

import numpy as np

from Project04.Evaluation.EvaluationCallable import EvaluationCallable
from Project04.NeuralNetwork import NeuralNetwork
import copy

class TuningUtility:

    def __init__(self, algorithm, folds, tuning_data, evaluation_method, network_parameters, population_size,
                 generations, hyperparameters, hyperparamters_tuning_order):
        self.algorithm = algorithm

        self.folds = folds

        self.tuning_data = tuning_data

        self.evaluation_method = evaluation_method

        self.network_parameters = network_parameters

        self.population_size = population_size
        self.generations = generations

        # Dictionary of hyperparameters.
        # Key = string name of hyperparameter
        # Value = [untuned_value, start_value, final_value, step_size]
        self.hyperparameters = hyperparameters
        self.tuning_order = hyperparamters_tuning_order

    def tune_hyperparameters(self):
        best_hp = copy.copy(self.hyperparameters)

        # Loop through the hyperparameters and add the initial value to the best_hp
        for hp in self.tuning_order:
            best_hp[hp] = self.hyperparameters[hp][0]

        # Tune each value one at a time in the given order
        for hp in self.tuning_order:
            untuned_value, start_value, final_value, step_size = self.hyperparameters[hp]

            # Keep track of the performances for each value
            performances = dict()

            # Loop through each of the possible values
            for value in np.arange(start_value, final_value, step_size):
                best_hp[hp] = value

                jobs = []
                manager = multiprocessing.Manager()
                fold_results = manager.dict()
                for fold in self.folds:
                    process = multiprocessing.Process(target=self.tune_single_fold, args=(id, best_hp, fold, fold_results))
                    jobs.append(process)
                    process.start()

                for j in jobs:
                    j.join()

                # Add the average performance of the hyperparameters to the dict
                performances[value] = sum(fold_results.values()) / len(self.folds)
                print("Finished hp: ", hp, " for value: ", value)
                print("Performance: ", performances[value])
            print("Best ", hp, ": ", best_hp[hp])
            # Keep the best hyperparameter
            best_hp[hp] = min(performances, key=performances.get)

        return best_hp

    def tune_single_fold(self, id, best_hp, fold, results):

        # Train the algorithm and get the best network and fitness
        network, fitness = self.train_on_fold(best_hp, fold)
        tuning_fitness = self.evaluation_method(self.tuning_data, network)
        results[id] = tuning_fitness

    def train_on_fold(self, best_hp, fold):
        # Create a population of networks
        networks = []
        for _ in range(self.population_size):
            networks.append(NeuralNetwork(self.network_parameters['shape'],
                                          self.network_parameters['output_transformer'],
                                          self.network_parameters['regression'],
                                          self.network_parameters['random_weight_range']))

        method = EvaluationCallable(fold, self.evaluation_method)
        alg = self.algorithm(networks, best_hp, method)
        # Train the algorithm and get the best network and fitness
        return alg.train(self.generations)
