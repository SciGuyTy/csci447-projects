from typing import Any, List, Tuple
import pandas as pd
from Evaluation.EvaluationMeasure import EvaluationMeasure as EM
from scipy.stats import ttest_ind
from scipy import stats
import numpy as np

class ExperimentHelper:
    @classmethod
    def convert_results_to_measures(cls, results: List[pd.DataFrame], classes: List[Any] = []):
    # def convert_results_to_measures(cls, results: List[dict[str, int]]):
        """Convert TP, TN, FP, FN to precision, recall, F1, and 0-1 loss scores

        results: List[dict[str, int]]
            The of list containing a count of true positives (TP), true negatives
            (TN), false positives (FP), and false negatives (FN) of each fold
            from the experiment.

        classes: List[Any]
            The classes to individually analyze as 'positive'

        Returns
        -------
        metrics: Dict[str, Dataframe]
            A dictionary containing DataFrames with the precision, recall, F1, and 0-1 loss scores for the data for each positive class
        """
        # Initialize a dictionary to store evaluation metrics for each class level
        metrics = dict.fromkeys(classes)

        # For each class level, compute the evaluation metrics on the folded data
        for classification in metrics:
            # Initialize output Dataframe
            measures = pd.DataFrame(
                data={"precision": [], "recall": [], "f1": [], "0-1": []}
            )

            # Loop through each fold
            for i, fold in enumerate(results):
                # Calculate each measure and add it to the dataframe
                    measures.loc[i] = [
                        EM.calculate_precision(fold, classification),
                        EM.calculate_recall(fold, classification),
                        EM.calculate_f_beta_score(fold, classification),
                        EM.calculate_0_1_loss(fold),
                    ]
            
            metrics[classification] = measures

        return metrics

    @classmethod
    def run_t_tests_on_columns(cls, measures_1: pd.DataFrame, measures_2: pd.DataFrame):
        """Run t-tests on each measure column

        measures_1: Dataframe
            The precision, recall, F1, and 0-1 loss scores for one run of the algorithm

        measures_2: Dataframe
            The precision, recall, F1, and 0-1 loss scores for another run of the algorithm

        Returns
        -------

        results: List[dict[str, float]]
            The of list containing a t-test results for each column in the two measures
        """
        # Get the set of columns to run the t-tests on.
        cols = set(measures_1.columns).intersection(set(measures_2.columns))

        # Initialize the results
        t_tests = dict()

        # Run the t-test on each column and record the t-test results
        for col in cols:
            print(col, ":", ExperimentHelper.run_ks_test(measures_1[col]))
            print(col, ":", ExperimentHelper.run_ks_test(measures_2[col]))
            t_tests[col] = ttest_ind(measures_1[col], measures_2[col])

        return t_tests

    @classmethod
    def format_results(
        cls,
        results: dict[str, dict[str, Tuple[Any, Any]]],
        index_labels: List[str] = ["t-statistic", "p-value"],
    ):
        """Print out the results for a given experiment

        results: dict[str, dict[str, Tuple[Any, Any]]]
            A dictionary which maps classes to a dictionary containing the results
        for said class

        index_labels: List[str]
            A list of labels that describe the respective rows in the results object.
            Defaults to `["t-statistic", "p-value"]`
        """
        # Represent each response object into a pandas DataFrame
        for result in results:
            results[result] = pd.DataFrame(results[result], index=index_labels)

        # Concatenate all results into a single DataFrame
        return pd.concat(results, axis=1).transpose()


            # num_true_positives = results[positive_class][positive_class]
            # num_actual = results[positive_class].sum()

            # return num_true_positives / num_actual

    @classmethod
    def run_ks_test(cls, column: pd.Series):
        """Return the KS Normality test results

                column: pd.Series
                    A column to be checked for normality
        """
        return stats.kstest(column, np.random.normal(column.mean(), column.std(), 1000))
