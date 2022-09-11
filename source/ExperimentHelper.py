from curses import panel
from typing import Any, List, Tuple
from unittest import result
import pandas as pd
from Evaluation.EvaluationMeasure import EvaluationMeasure as EM
from scipy.stats import ttest_ind


class ExperimentHelper:
    @classmethod
    def convert_results_to_measures(cls, results: List[dict[str, int]]):
        """Convert TP, TN, FP, FN to precision, recall, F1, and 0-1 loss scores

        results: List[dict[str, int]]
            The of list containing a count of true positives (TP), true negatives
            (TN), false positives (FP), and false negatives (FN) of each fold
            from the experiment.

        Returns
        -------
        measures: Dataframe
            The precision, recall, F1, and 0-1 loss scores for the data
        """
        # Initialize output Dataframe
        measures = pd.DataFrame(
            data={"precision": [], "recall": [], "f1": [], "0-1": []}
        )

        # Loop through each fold
        for i, fold in enumerate(results):
            # Calculate each measure and add it to the dataframe
            measures.loc[i] = [
                EM.calculate_precision(fold),
                EM.calculate_recall(fold),
                EM.calculate_f_beta_score(fold),
                EM.calculate_0_1_loss(fold),
            ]

        return measures

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
        return pd.concat(results, axis=1)
