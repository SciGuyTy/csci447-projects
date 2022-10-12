from typing import Any, Iterable
from unittest import result
from source.Evaluation.EvaluationMeasure import EvaluationMeasure
from source.Evaluation.CrossValidation import CrossValidation
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Algorithms.DistanceFunctions.DistanceFunction import DistanceFunction
from source.Algorithms.KNN import KNN
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from itertools import count

counter = count()
class EditedKNN(KNN):
    def __init__(
        self,
        training_data: pd.DataFrame,
        target_feature: str,
        regression=False,
        sigma: float = None,
        epsilon: float = None,
        k: int = 1,
        cluster=False,

    ):
        """
        Perform edited k-NN by minimizing (editing) the training data through excision
        of either noisy data or redundant data

        Parameters:
        -----------
        training_data: pd.DataFrame
            The training data

        tuning_data: pd.DataFrame
            The tuning data (used to evaluate performance for each fold of the editing
            process)

        target_feature: str
            The target ("response") feature

        regression: bool
            Whether the training data is a regression dataset or classification dataset
            (Defaults to false)

        sigma: float
            The threshold value for the Gaussian kernel

        distance_function: DistanceFunction
            The distance function to use when determining the k nearest neighbors
            (Defaults to the Minkowski metric of order 2)
        """
        # Extend the KNN class
        super().__init__(
            training_data, target_feature, regression, sigma,
        )

        self.epsilon = epsilon
        self.k = k

    def _predict_response(self, instance: pd.Series, k: int) -> Any:
        """
        Predict the response value for a novel data point

        Parameters:
        -----------
        k: int
            The number of nearest neighbors to consider when making
            the prediction

        Returns:
        --------
        The prediction for the response value of the novel instance
        """

        # Find the k nearest neighbors
        neighbors_distance = self.find_k_nearest_neighbors(instance, k)
        if self.regression:
            # Perform regression and return the response
            return self.regress(neighbors_distance)
        else:
            # Apply a vote of plurality and return the response
            neighbors = [i[1] for i in neighbors_distance]
            return self.vote(neighbors)

    def _minimize_data(
        self, k: int, reduce_redundancy: bool, err_threshold: float
    ) -> pd.DataFrame:
        """
        Remove samples based on a criteria. Removing samples that produce incorrect
        predictions will serve to reduce noise ipn the data. Removing samples that produce
        correct predictions will serve to reduce redundant data

        Parameters
        ----------
        k: int
            The number of clusters with which to fit

        reduce_redundancy: bool
            Whether the minimization process should target the removal of redundant
            data, or whether it should target the removal of noisy data

        err_threshold: float
            The threshold around the actual value for a given instance
            with which a predicted response value from a regression model
            will still be considered 'correct'

        Returns
        -------
        A DataFrame containing the minimized dataset
        """
        # List to keep track of samples that should be dropped from the training data
        samples_to_drop = []

        for index, instance in self.training_data.iterrows():
            # Drop the current instance from the training data
            self.training_data.drop(index, inplace=True)

            # Retrieve the relevant features for the given instance
            sample_vector = instance[self.features]

            # Predict the response value for the given sample
            prediction = self._predict_response(sample_vector, k)

            correct_response = instance[self.target_feature]

            # Boolean representing if an instance was correctly predicted
            correctly_predicted = False

            # For regression data, check if the predicted response is within
            # the error threshold for the actual response
            if self.regression and (
                prediction >= correct_response - err_threshold
                or prediction <= correct_response + err_threshold
            ):
                correctly_predicted = True
            elif not self.regression:
                correctly_predicted = correct_response == prediction

            if reduce_redundancy and correctly_predicted:
                # Remove redundant data
                samples_to_drop.append(index)
            elif not reduce_redundancy and not correctly_predicted:
                # Remove noisy data
                samples_to_drop.append(index)

            # Reinsert the instance into the training data
            self.training_data.loc[index] = instance

        # Drop irrelevant samples from the training data
        self.training_data.drop(index=samples_to_drop, inplace=True)

    def _check_performance(self, k) -> float:
        """
        Check the performance of the model on a training set

        Parameters:
        k: int
            The number of neighbors to consider when performing knn
        """
        # Compute the performance metric for each fold from CV
        if self.regression:
            results = pd.DataFrame(columns=["actual", "predicted"])

            for index, sample in self.training_data.iterrows():
                # Predict the response value for the given sample
                prediction = self._predict_response(sample, k)

                # Store results for prediction
                results.loc[index] = [sample[self.target_feature], prediction]

            # Return the performance of the model trained with the current training data
            return EvaluationMeasure.calculate_means_square_error(results)
        else:
            # Get a list of all class levels
            classes = self.training_data[self.target_feature].unique()

            # Results for this fold
            results = pd.DataFrame(0, columns=classes, index=classes)

            for _, sample in self.training_data.iterrows():
                # Predict the response value for the given sample
                prediction = self._predict_response(sample, k)
                actual = sample[self.target_feature]

                # Store results for prediction in confusion matrix
                results[actual][prediction] += 1

            #print("Ending check performance for", k)
            # Return the performance of the model trained with the current training data
            return EvaluationMeasure.calculate_0_1_loss(results)
    
    def predict(
        self,
        instance: Iterable,
        k: int,
        reduce_redundancy=False,
        err_threshold=0.0,
    ) -> str:
        """
        Edit the underlying dataset and preform a prediction on the edited dataset

        instance: Iterable
            The instance to make a prediction for

        k: int
            The number of neighbors to consider when making a prediction

        reduce_redundancy: bool
            Whether the minimization process should target the removal of redundant
            data, or whether it should target the removal of noisy data (defaults to
            False... i.e., it targets removal of noisy data by default)

        err_threshold: float
            The threshold around the actual value for a given instance
            with which a predicted response value from a regression model
            will still be considered 'correct'
        """
        self.train(k, reduce_redundancy=reduce_redundancy)

        # Report the prediction based on the minimized dataset
        return self._predict_response(instance, k)

    def train(self, k: int, reduce_redundancy=False):

        # Initialize the previous edit and performance to that of the models performance
        # on unedited data
        previous_edit = self.training_data
        previous_performance = current_performance = self._check_performance(self.k)

        # Continue to minimize the training data if the performance until the performance
        # of the model stops improving
        count = 1
        while True:
            # Minimize the data
            self._minimize_data(k, reduce_redundancy, self.epsilon)
            #print("k {} finished minimizing {}".format(k, count))

            # Update the current performance
            current_performance = self._check_performance(k)
            count += 1
            if current_performance > previous_performance:
                # Update the previous performance/data
                previous_performance = current_performance
                previous_edit = self.training_data
            else:
                break
        #print("K {} done after {} loops".format(k, count))
        # Set the training data to that of the previous edit
        self.training_data = previous_edit


