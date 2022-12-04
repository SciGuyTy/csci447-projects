import multiprocessing

import numpy as np

from Project04.Evaluation.EvaluationCallable import EvaluationCallable
from Project04.NeuralNetwork import NeuralNetwork
import copy

from Project04.Utilities.Utilities import Utilities


class TuningUtility:

    def __init__(self, algorithm, training_test_folds, tuning_data, evaluation_method, network_parameters, population_size,
                 generations, hyperparameters, hyperparamters_tuning_order):
        self.algorithm = algorithm

        self.training_test_folds = training_test_folds
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
                for i, (training_data, hold_out, norm_params) in enumerate(self.training_test_folds):
                    process = multiprocessing.Process(target=self.tune_single_fold, args=(i, best_hp, training_data, norm_params, fold_results))
                    jobs.append(process)
                    process.start()

                for j in jobs:
                    j.join()

                # Add the average performance of the hyperparameters to the dict
                performances[value] = sum(fold_results.values()) / len(self.training_test_folds)
                print("Finished hp: ", hp, " for value: ", value)
                print("Performance: ", performances[value])
            print("Best ", hp, ": ", best_hp[hp])
            # Keep the best hyperparameter
            best_hp[hp] = min(performances, key=performances.get)

        return best_hp

    def tune_single_fold(self, i, best_hp, training_data, norm_params, results):

        # Train the algorithm and get the best network and fitness
        network, fitness = self.train_on_fold(best_hp, training_data)
        tuning_data = Utilities.normalize_set_by_params(self.tuning_data.copy(), norm_params)
        tuning_fitness = self.evaluation_method(tuning_data, network)
        print(f"{i=}, {fitness=}, {tuning_fitness=} ")
        results[i] = tuning_fitness

    def train_on_fold(self, best_hp, fold, best_networks=None, id=None):
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
        network, fitness = alg.train(self.generations)
        if best_networks is not None and id is not None:
            best_networks[id] = network
        return network, fitness