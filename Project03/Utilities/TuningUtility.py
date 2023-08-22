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

        # Use 10 minibatches
        self.minibatch_size = training_params['batch_size']

        self.iterations = training_params['epochs']

    def tune_for_h_hidden_layers(self, h):
        shapes = self._get_all_shapes_for_hidden_layers(h)
        print("Trying {} shapes".format(len(shapes)))
        jobs = []
        manager = multiprocessing.Manager()
        fold_results = manager.dict()
        for id, (training_data, hold_out, norm_params) in enumerate(self.folds):
            process = multiprocessing.Process(target=self.tune_single_fold, args=(id, shapes, training_data, hold_out,
                                                                                  norm_params, fold_results))
            jobs.append(process)
            process.start()

        for j in jobs:
            j.join()
        return fold_results.values()

    def tune_single_fold(self, id, shapes, training_data, testing_data, norm_params, fold_results):
        cost_for_shapes = dict()
        tuning_data = Utilities.normalize_set_by_params(self.tuning_data.copy(), norm_params)
        for shape in shapes:
            print("Fold: {}, shape: {}".format(id, shape))
            nn = NeuralNetwork(shape, self.output_transformer, not self.classification)
            nn.train(training_data, self.target_column, self.learning_rate, self.momentum, self.iterations,
                     self.minibatch_size, self.target_modifer)

            cv = CrossValidation(tuning_data, self.target_column, not self.classification)
            results_for_model_on_fold = cv.calculate_results_for_fold(nn, testing_data)
            tuning_results = cv.calculate_results_for_fold(nn, tuning_data)
            if self.classification:
                cost = EvaluationMeasure.calculate_0_1_loss(tuning_results)
                #cost2 = EvaluationMeasure.calculate_0_1_loss(results_for_model_on_fold)
            else:
                cost = EvaluationMeasure.calculate_means_square_error(tuning_results)
                #cost2 = EvaluationMeasure.calculate_means_square_error(results_for_model_on_fold)
            cost_for_shapes[tuple(shape)] = (cost, shape, nn)

        if self.classification:
            fold_results[id] = (max(cost_for_shapes.values()))
        else:
            fold_results[id] = (min(cost_for_shapes.values()))
    def _get_all_shapes_for_hidden_layers(self, num_hidden_layers):
        shapes = self._generate_possible_configs(self.num_inputs, self.num_outputs, num_hidden_layers)
        for i in range(len(shapes)):
            shapes[i] = [self.num_inputs] + shapes[i] + [self.num_outputs]

        return shapes

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
