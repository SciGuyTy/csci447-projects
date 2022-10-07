import math

from Project02.source.Evaluation.CrossValidation import CrossValidation
from Project02.source.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project02.source.Utilities.Utilities import Utilities
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


class TuningUtility:

    def __init__(self, model, data: pd.DataFrame, target_feature="class", regression=False, model_params=[]):
        self.model = model
        self.data = data
        self.regression = regression
        self.model_params = model_params
        self.target_feature = target_feature
        self.CV = CrossValidation(self.data, regression=self.regression, target_feature=self.target_feature)

    def tune_k_for_folds(self, training_test_data, tuning_data:pd.DataFrame, model_params=[], k_range=None):
        if k_range is None:
            # TODO Might have to change this len() to be the number of rows in edited dataset
            k_range = (1, math.ceil(len(training_test_data[0][0].index)**.5))

        best_ks = []
        best_results = dict()
        for fold, (training_data, _, norm_params) in enumerate(training_test_data, start=1):
            results = self.tune_k_for_single_fold(training_data, tuning_data.copy(), norm_params=norm_params, model_params=model_params, k_range=k_range)
            if self.regression:
                results_df = pd.DataFrame(results.items(), columns=['k', 'MSE'])
                results_df = results_df.sort_values(by=['MSE', 'k'], ascending=[True, True], ignore_index=True)
            else:
                results_df = pd.DataFrame(results.items(), columns=['k', '0/1'])
                results_df = results_df.sort_values(by=['0/1', 'k'], ascending=[False, True], ignore_index=True)
            k, measure = results_df.loc[0]
            best_results[fold] = (k, measure)
            best_ks.append(k)
            k_range = TuningUtility.get_new_range(best_ks, 5)
            print("Best ks", best_ks)
            print("current range: ", k_range)
        return best_results

    @staticmethod
    def get_new_range(items, margin):
        new_min = max(0, min(items) - margin)
        new_max = max(items) + margin
        return int(new_min), int(new_max)

    def tune_k_for_single_fold(self, training_data: pd.DataFrame, test_data: pd.DataFrame, norm_params: pd.DataFrame = None, model_params=[], k_range=None):
        if k_range is None:
            # TODO Might have to change this len() to be the number of rows in edited dataset
            k_range = (1, math.ceil(len(training_data.index)**.5))

        if norm_params is not None:
            test_data = Utilities.normalize_set_by_params(test_data, norm_params)

        if not self.regression:
            algorithm = self.model(training_data, self.target_feature, *model_params)

            print("k_range", k_range)

            jobs = []

            manager = multiprocessing.Manager()
            results = manager.dict()
            for k in range(k_range[0], k_range[1] + 1):
                process = multiprocessing.Process(target=TuningUtility.get_01_loss_for_k, args=(algorithm, self.CV, test_data, k, results))
                jobs.append(process)
                process.start()

            for j in jobs:
                j.join()

        else:
            algorithm = self.model(training_data, self.target_feature, *model_params)
            print("k_range", k_range)

            jobs = []

            manager = multiprocessing.Manager()
            results = manager.dict()
            for k in range(k_range[0], k_range[1] + 1):
                process = multiprocessing.Process(target=TuningUtility.get_mean_squared_error_for_k,
                                                  args=(algorithm, self.CV, test_data, k, results))
                jobs.append(process)
                process.start()

            for j in jobs:
                j.join()


        return results


    @staticmethod
    def get_01_loss_for_k(algorithm, cv, test_data, k, results):
        print("k: ", k)
        if k < 1:
            results[k] = None
            return
        fold_results = cv.calculate_results_for_fold(algorithm, test_data, predict_params=[k])
        loss = EvaluationMeasure.calculate_0_1_loss(fold_results)
        results[k] = loss
        print(results)

    @staticmethod
    def get_mean_squared_error_for_k(algorithm, cv, test_data, k, results):
        print("k: ", k)
        if k < 1:
            results[k] = None
            return
        fold_results = cv.calculate_results_for_fold(algorithm, test_data, predict_params=[k])
        MSE = EvaluationMeasure.calculate_means_square_error(fold_results)
        results[k] = MSE
        print(results)