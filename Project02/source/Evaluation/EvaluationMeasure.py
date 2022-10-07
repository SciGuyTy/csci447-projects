import pandas as pd

class EvaluationMeasure:
    @classmethod
    def calculate_precision(cls, results: pd.DataFrame, positive_class: str) -> float:
        """Calculates the precision based on the results

        Parameters
        ----------
        results: pd.DataFrame
            The results from the experiment

        positive_class: str
            The positive class in the calculation

        Returns
        -------
        A float representing the recall of the provided results
        """

        num_true_positives = results[positive_class][positive_class]
        num_predicted_positives = results.sum(axis=1)[positive_class]

        # All predictions are negative, and 0 positive predictions were reported 
        # so the precision approaches 1
        if num_predicted_positives == 0:
            return 1

        return num_true_positives / num_predicted_positives

    @classmethod
    def calculate_recall(cls, results: pd.DataFrame, positive_class: str) -> float:
        """Calculates the recall based on the results

        Parameters
        ----------
        results: pd.DataFrame
            The results from the experiment

        positive_class: str
            The positive class in the calculation

        Returns
        -------
        A float representing the recall of the provided results
        """

        num_true_positives = results[positive_class][positive_class]
        num_predicted_positives = results.sum(axis=0)[positive_class]

        if num_predicted_positives == 0:
            return 1

        return num_true_positives / num_predicted_positives

    @classmethod
    def calculate_f_beta_score(
        cls, results: pd.DataFrame, positive_class: str, beta: int = 1
    ) -> float:
        """Calculates the F_Beta score based on the results and the beta value

        Parameters
        ----------
        results: pd.DataFrame
            The results from the experiment

        (Optional) beta: int
            The beta value to use when calculating the F_Beta score. Default is 1

        positive_class: str
            The positive class in the calculation

        Returns
        -------
        A float representing the f-beta score for the provided results
        """
        precision = cls.calculate_precision(results, positive_class)
        recall = cls.calculate_recall(results, positive_class)
        denom = precision * beta**2 + recall

        if denom == 0:
            return 0.0

        return ((1 + beta**2) * precision * recall) / denom

    @classmethod
    def calculate_0_1_loss(cls, results: pd.DataFrame) -> float:
        """Calculates the 0-1 loss measure based on the results

        Parameters
        ----------
        results: dict[str, int]
            The results from the experiment

        Returns
        -------
        A float representing the 0_1 loss for the provided results
        """
        num_correct_predictions = 0

        # Compute the total number of correct predictions (TP for each class level)
        for column in results.columns:
            num_correct_predictions += results[column][column]

        return num_correct_predictions / results.to_numpy().sum()

    @classmethod
    def calculate_means_square_error(cls, results: pd.DataFrame) -> float:
        """
        Calculate mean square error for a set of predictions

        Parameters
        -----------
        results: pd.DataFrame
            The results from the experiment, with a colum for the actual response
            and a model for the predicted response

        Returns
        -------
        A float representing the mean square error for the provided results
        """
        # Sum of square error for each result in the set
        total_error = 0

        # For each result in the set, compute the square error and add it to the total_error
        for index, result in results.iterrows():
            total_error += (result["actual"] - result["predicted"]) ** 2

        # Divide the total_error by the number of results in the set to compute and return
        # the mean square error
        return total_error / len(results.index)
