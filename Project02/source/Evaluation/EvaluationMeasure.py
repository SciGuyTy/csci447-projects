import re
import pandas as pd


class EvaluationMeasure:
    @classmethod
    def calculate_precision(cls, results: pd.DataFrame, positive_class: str):
        """Calculates the precision based on the results

        Parameters
        ----------
        results: pd.DataFrame
            The results from the experiment

        positive_class: str
            The positive class in the calculation
        """

        num_true_positives = results[positive_class][positive_class]
        num_predicted_positives = results.sum(axis=1)[positive_class]

        # All predictions are negative, and 0 positive predictions were reported so the precision approaches 1
        if num_predicted_positives == 0:
            return 1

        return num_true_positives / num_predicted_positives


    @classmethod
    def calculate_recall(cls, results: pd.DataFrame, positive_class: str):
        """Calculates the recall based on the results

        Parameters
        ----------
        results: pd.DataFrame
            The results from the experiment

        positive_class: str
            The positive class in the calculation
        """

        num_true_positives = results[positive_class][positive_class]
        num_predicted_positives = results.sum(axis=0)[positive_class]

        # No positive cases in the input data, and 0 positive classes were predicted so the recall is 1?
        if num_predicted_positives == 0:
            return 1

        return num_true_positives / num_predicted_positives

    @classmethod
    def calculate_f_beta_score(cls, results: pd.DataFrame, positive_class: str, beta: int = 1):
        """Calculates the F_Beta score based on the results and the beta value

        Parameters
        ----------
        results: pd.DataFrame
            The results from the experiment

        (Optional) beta: int
            The beta value to use when calculating the F_Beta score. Default is 1

        positive_class: str
            The positive class in the calculation
        """
        precision = cls.calculate_precision(results, positive_class)
        recall = cls.calculate_recall(results, positive_class)
        denom = precision * beta**2 + recall
        
        if denom == 0:
            return 0.0

        return ((1 + beta**2) * precision * recall) / denom

    @classmethod
    def calculate_0_1_loss(cls, results: pd.DataFrame):
        """Calculates the 0-1 loss measure based on the results

        Parameters
        ----------
        results: dict[str, int]
            The results from the experiment
        """
        num_correct_predictions = 0

        # Compute the total number of correct predictions (TP for each class level)
        for column in results.columns:
            num_correct_predictions += results[column][column]

        return num_correct_predictions / results.to_numpy().sum()

    @classmethod
    def calculate_square_loss(cls, results: pd.DataFrame):
        """
        Calculate square loss for a regression prediction

        Parameters:
        -----------
        results: pd.DataFrame
            The results from the experiment, with a colum for the actual response
            and a model for the predicted response
        """
        return (results["actual"] - results["predicted"])**2
