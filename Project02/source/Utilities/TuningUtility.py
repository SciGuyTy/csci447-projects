import math
import time
import copy
from Project02.source.Evaluation.CrossValidation import CrossValidation
from Project02.source.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project02.source.Utilities.Utilities import Utilities
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import numpy


class TuningUtility:

    def __init__(self, model, data: pd.DataFrame, target_feature="class", regression=False):
        # The algorithm to use (KNN, EKNN, K-Means)
        self.model = model

        # The data set
        self.data = data

        # A boolean representing whether this is a regression problem
        self.regression = regression

        # The column name of the response field
        self.target_feature = target_feature

        # The Cross Validation utility
        self.CV = CrossValidation(self.data, regression=self.regression, target_feature=self.target_feature)

    def tune_sigma_k_and_epsilon_for_folds(self, training_test_data, tuning_data, sigma_range, sigma_step, epsilon_range, epsilon_step, k_range=None):
        # Initialize the dictionary of all results.
        # Key = (epsilon, sigma)
        # Value = list of data frames containing the k, mse, and trained model
        all_results = dict()
        # Perform a grid search over epsilon and sigma
        for epsilon in numpy.arange(epsilon_range[0], epsilon_range[1]+epsilon_step, epsilon_step):
            for sigma in numpy.arange(sigma_range[0], sigma_range[1]+sigma_step, sigma_step):
                print(f"{sigma=},{epsilon=}")
                # Calculate the results and add them to the dictionary
                results = self.tune_k_for_folds(training_test_data, tuning_data, model_params=[sigma, epsilon], include_k_in_model=True, k_range=k_range)
                all_results[(epsilon, sigma)] = results
        return all_results
    def tune_sigma_and_k_for_folds(self, training_test_data, tuning_data: pd.DataFrame, sigma_range, sigma_step, epsilon=None):
        # Initialize the dictionary of all results.
        # Key = sigma
        # Value = list of data frames containing the k, mse, and trained model
        all_results = dict()
        # Search through the values of sigma
        for sigma in numpy.arange(sigma_range[0], sigma_range[1]+1, sigma_step):
            print("Sigma {}".format(sigma))
            # Calculate the results for each sigma and k and add them to the dictionary
            results = self.tune_k_for_folds(training_test_data, tuning_data, model_params=[sigma])
            all_results[sigma] = results
        return all_results

    @staticmethod
    def get_best_parameters_and_results(all_results):
        # Create a dictionary of the folds
        fold_sigma_k = dict()
        # Transform the data into a dictionary of a list of dictionaries for the folds and results.
        for sigma, values in all_results.items():
            for fold, (k, mse, model) in values.items():
                if fold not in fold_sigma_k:
                    fold_sigma_k[fold] = []
                fold_sigma_k[fold].append({'k': k, 'sigma': sigma, 'mse': mse, 'model': model})

        # Transform the dictionary into just the best performing result for each fold
        for fold in fold_sigma_k:
            fold_sigma_k[fold] = min(fold_sigma_k[fold], key=lambda item: item['mse'])

        return fold_sigma_k


    def tune_k_for_folds(self, training_test_data, tuning_data:pd.DataFrame, model_params=[], k_range=None, include_k_in_model=False):
        # If there isn't a defined range for the k values, use a default baseline
        if k_range is None:
            # [1, sqrt(n)]
            k_range = (1, math.ceil(len(training_test_data[0][0].index)**.5))

        best_ks = []
        best_results = dict()
        start_time = time.time()

        # Loop through the folds and record the hyperparameters, results, and the trained model (K, MSE/0-1 loss, Model).
        for fold, (training_data, _, norm_params) in enumerate(training_test_data, start=1):
            # Calculate the results for this fold
            results = self.tune_k_for_single_fold(training_data, tuning_data.copy(), norm_params=norm_params, model_params=model_params, k_range=k_range, train=True)
            # Transform the data, label the columns, and sort them based on the measure. Ties are broken by the lower k value winning
            results_df = pd.DataFrame(results).T
            results_df.reset_index(inplace=True)
            if self.regression:
                results_df.columns=['k', 'MSE', 'Model']
                results_df = results_df.sort_values(by=['MSE', 'k'], ascending=[True, True], ignore_index=True)
            else:
                results_df.columns=['k', '0/1', 'Model']
                results_df = results_df.sort_values(by=['0/1', 'k'], ascending=[False, True], ignore_index=True)
            # Record the best performing results.
            k, measure, model = results_df.loc[0]
            best_results[fold] = (k, measure, model)
            best_ks.append(k)

            # Update the range of k values we search for the next fold.
            k_range = TuningUtility.get_new_range(best_ks, 5)
            print("Best ks", best_ks)
            print("current range: ", k_range)
        print("time to tune folds", time.time()-start_time)
        return best_results

    @staticmethod
    def get_new_range(items, margin):
        # Create a new range by taking the minimum and
        # maximum of the best k's and extending that range
        # by the margin on each end.
        new_min = max(0, min(items) - margin)
        new_max = max(items) + margin
        return int(new_min), int(new_max)

    def tune_k_for_single_fold(self, training_data: pd.DataFrame, test_data: pd.DataFrame, norm_params: pd.DataFrame = None, model_params=[], k_range=None, train=None):
        # If there isn't a defined range for the k values, use a default baseline
        if k_range is None:
            # [1, sqrt(n)]
            k_range = (1, math.ceil(len(training_data.index)**.5))

        # Normalize the data if we have that ability
        if norm_params is not None:
            test_data = Utilities.normalize_set_by_params(test_data, norm_params)

        if not self.regression:
            print("k_range", k_range)
            jobs = []
            manager = multiprocessing.Manager()
            # A dictionary proxy to handle the returned results from the multi-processing.
            results = manager.dict()
            # Spawn a new process to run validation again the tuning data for each k value we want to examine
            for k in range(k_range[0], k_range[1] + 1):
                algorithm = self.model(training_data, self.target_feature, False, *model_params)
                args = [algorithm, self.CV, test_data, k, results]
                # If the algorithm needs to be trained, have the new process train it
                if train:
                    args += [algorithm.train, [k]]
                process = multiprocessing.Process(target=TuningUtility.get_01_loss_for_k, args=args)
                jobs.append(process)
                process.start()

        else:
            print("k_range", k_range)
            jobs = []
            manager = multiprocessing.Manager()
            # A dictionary proxy to handle the returned results from the multi-processing.
            results = manager.dict()
            # Spawn a new process to run validation again the tuning data for each k value we want to examine
            for k in range(k_range[0], k_range[1] + 1):
                algorithm = self.model(training_data, self.target_feature, True, *model_params)
                args = [algorithm, self.CV, test_data, k, results]
                # If the algorithm needs to be trained, have the new process train it
                if train:
                    args += [algorithm.train, [k]]
                process = multiprocessing.Process(target=TuningUtility.get_mean_squared_error_for_k,
                                                  args=args)
                jobs.append(process)
                process.start()

        # Pause the main thread until the k-values have all been evaluated
        for j in jobs:
            j.join()

        print(results)
        # Convert the dictionary proxy into a dictionary
        return dict(results)


    @staticmethod
    def get_01_loss_for_k(algorithm, cv, test_data, k, results, train_callable=None, train_params=[]):
        # Don't run on k values less than one
        if k < 1:
            results[k] = None
            return
        #print("k: ", k)
        # Train the model if possible
        if train_callable is not None:
            train_callable(*train_params)
        # Calculate the results for the specific hyperparameters
        fold_results = cv.calculate_results_for_fold(algorithm, test_data, predict_params=[k])
        # Calculate the loss from the results
        loss = EvaluationMeasure.calculate_0_1_loss(fold_results)
        # Add the results to the dictionary proxy
        results[k] = (loss, algorithm)
        #print(results)

    @staticmethod
    def get_mean_squared_error_for_k(algorithm, cv, test_data, k, results, train_callable=None, train_params=[]):
        # Don't run on k values less than one
        if k < 1:
            results[k] = None
            return
        #print("k: ", k)
        # Train the model if possible
        if train_callable is not None:
            train_callable(*train_params)
        # Calculate the results for the specific hyperparameters
        fold_results = cv.calculate_results_for_fold(algorithm, test_data, predict_params=[k])
        # Calculate the MSE from the results
        MSE = EvaluationMeasure.calculate_means_square_error(fold_results)
        # Add the results to the dictionary proxy
        results[k] = (MSE, algorithm)
        #print(results)