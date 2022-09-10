import re
import pandas as pd


class EvaluationMeasure:
    @classmethod
    def calculate_precision(cls, results: dict[str, int]):
        """Calculates the precision based on the results

        Parameters
        ----------
        results: dict[str, int]
            The results from the experiment
        """
        try:
            return results["TP"] / (results["TP"] + results["FP"])
        except ZeroDivisionError:
            # All predictions are negative, and 0 positive predictions were reported so the precision is 1?
            return 1.0

    @classmethod
    def calculate_recall(cls, results: dict[str, int]):
        """Calculates the recall based on the results

        Parameters
        ----------
        results: dict[str, int]
            The results from the experiment
        """
        try:
            return results["TP"] / (results["TP"] + results["FN"])
        except ZeroDivisionError:
            # No positive cases in the input data, and 0 positive classes were predicted so the recall is 1?
            return 1.0

    @classmethod
    def calculate_f_beta_score(cls, results: dict[str, int], beta: int = 1):
        """Calculates the F_Beta score based on the results and the beta value

        Parameters
        ----------
        results: dict[str, int]
            The results from the experiment

        (Optional) beta: int
            The beta value to use when calculating the F_Beta score. Default is 1
        """
        precision = cls.calculate_precision(results)
        recall = cls.calculate_recall(results)
        try:
            return ((1 + beta**2) * precision * recall) / (
                precision * beta**2 + recall
            )
        except ZeroDivisionError:
            # If both precision and recall are 0, the F-Beta score is 0
            return 0.0

    @classmethod
    def calculate_0_1_loss(cls, results: dict[str, int]):
        """Calculates the 0-1 loss measure based on the results

        Parameters
        ----------
        results: dict[str, int]
            The results from the experiment
        """
        return (results["TP"] + results["TN"]) / sum(results.values())
