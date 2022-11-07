import math
import multiprocessing

import pandas as pd
from typing import Callable
from Project03.NeuralNetwork import NeuralNetwork

from Project03.Evaluation.CrossValidation import CrossValidation
from Project03.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project03.Utilities.Utilities import Utilities


class TuningUtility:

    def __init__(self, folds, tuning_data, target_column, num_inputs, num_outputs, target_modifer: Callable, output_transformer: Callable, classification: bool, training_params: dict):
        self.folds = folds
        self.tuning_data = tuning_data
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.target_modifer = target_modifer
        self.output_transformer = output_transformer
        self.target_column = target_column
        self.classification = classification

        self.training_params = training_params
        self.learning_rate = training_params['learning_rate']
        self.momentum = training_params['momentum']
        self.initial_weight_range = training_params['initial_weight_range']

        # Use 10 minibatches
        self.minibatch_size = training_params['batch_size']

        self.iterations = training_params['epochs']
        self.restricted_models = training_params.get('restrict_shapes', False)

    # Tune the models for h hidden layers
    def tune_for_h_hidden_layers(self, h):
        # Get all the shapes
        shapes = self._get_all_shapes_for_hidden_layers(h, self.restricted_models)
        print("Trying {} shapes".format(len(shapes)))

        # Store all the jobs and run each fold in its own process
        jobs = []
        manager = multiprocessing.Manager()
        fold_results = manager.dict()
        for id, (training_data, hold_out, norm_params) in enumerate(self.folds):
            process = multiprocessing.Process(target=self.tune_single_fold, args=(id, shapes, training_data, hold_out,
                                                                                  norm_params, fold_results))
            jobs.append(process)
            process.start()

        # Rejoin the jobs into one process
        for j in jobs:
            j.join()

        # Return the results for each fold
        return fold_results.values()

    # Tune a single fold (set up to allow multiprocessing
    def tune_single_fold(self, id, shapes, training_data, testing_data, norm_params, fold_results):
        cost_for_shapes = dict()
        # Normalize the tuning data
        tuning_data = Utilities.normalize_set_by_params(self.tuning_data.copy(), norm_params)

        # Tune the model for each shape
        for shape in shapes:
            print("Fold: {}, shape: {}".format(id, shape))
            nn = NeuralNetwork(shape, self.output_transformer, not self.classification, self.initial_weight_range)
            nn.train(training_data, self.target_column, self.learning_rate, self.momentum, self.iterations,
                     self.minibatch_size, self.target_modifer)

            cv = CrossValidation(tuning_data, self.target_column, not self.classification)

            # Determine the performance on the tuning data
            tuning_results = cv.calculate_results_for_fold(nn, tuning_data)
            print(tuning_results)

            # Calculate the proper result
            if self.classification:
                cost = EvaluationMeasure.calculate_0_1_loss(tuning_results)
            else:
                cost = EvaluationMeasure.calculate_means_square_error(tuning_results)
            cost_for_shapes[tuple(shape)] = (cost, shape, nn)

        # return the best model
        if self.classification:
            fold_results[id] = (max(cost_for_shapes.values()))
        else:
            fold_results[id] = (min(cost_for_shapes.values()))

    # Gets all the shapes according to the restriction rules
    def _get_all_shapes_for_hidden_layers(self, num_hidden_layers, restricted):
        if not restricted:
            shapes = self._generate_possible_configs(self.num_inputs, self.num_outputs, num_hidden_layers)
            for i in range(len(shapes)):
                shapes[i] = [self.num_inputs] + shapes[i] + [self.num_outputs]

            return shapes
        else:
            shapes = []
            for i in range(self.num_outputs, self.num_inputs + 1):
                shape = []
                shape += [self.num_inputs]
                shape += [i for _ in range(num_hidden_layers)]
                shape += [self.num_outputs]
                shapes.append(shape)
            return shapes

    # Recursive method for obtaining all the shapes
    @staticmethod
    def _generate_possible_configs(num_of_inputs, num_of_outputs, num_of_hidden_layers):
        if num_of_hidden_layers == 0:
            return [[]]
        if num_of_hidden_layers == 1:
            return [[i] for i in range(num_of_outputs, num_of_inputs + 1)]
        output = []
        for i in range(num_of_outputs, num_of_inputs + 1):
            x = TuningUtility._generate_possible_configs(i, num_of_outputs, num_of_hidden_layers - 1)
            for j in x:
                output.append([i] + j)
        return output
