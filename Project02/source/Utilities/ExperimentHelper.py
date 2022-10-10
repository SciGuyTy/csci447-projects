from typing import Any, List, Tuple
import pandas as pd
from source.Evaluation.EvaluationMeasure import EvaluationMeasure as EM

class ExperimentHelper:
    @classmethod
    def convert_classification_results_to_measures(cls, results: List[pd.DataFrame], classes: List[Any] = []):
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
    def convert_regression_results_to_measures(cls, results: List[pd.DataFrame]):
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
        # Initialize output Dataframe
        measures = pd.DataFrame(
            data={"MSE": []}
        )

        # Loop through each fold
        for i, fold in enumerate(results):
            # Calculate each measure and add it to the dataframe
                measures.loc[i] = [
                    EM.calculate_means_square_error(fold),
                ]
        
        return measures