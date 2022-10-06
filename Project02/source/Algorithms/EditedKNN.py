from source.Evaluation.EvaluationMeasure import EvaluationMeasure
from source.Evaluation.CrossValidation import CrossValidation
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Algorithms.DistanceFunctions.DistanceFunction import DistanceFunction
from source.Algorithms.KNN import KNN
import pandas as pd

class EditedKNN(KNN):
    def __init__(
        self, 
        training_data: pd.DataFrame, 
        target_feature: str,
        regression=False, 
        h=None, 
        sigma=None, 
        distance_function: 
        DistanceFunction = Minkowski(),
        remove_noise = True
    ):
        super().__init__(training_data, target_feature, regression, h, sigma, distance_function)
        self.remove_noise = remove_noise

    def _classify(self, instance, k) -> str:
        neighbors_distance = self.find_k_nearest_neighbors(instance, k)

        if self.regression:
            return self.regress(neighbors_distance)
        else:
            neighbors = [i[1] for i in neighbors_distance]
            return self.vote(neighbors)

    def _minimize_data(self, k) -> pd.DataFrame:
        """
        Remove samples based on a criteria. Removing samples that produce incorrect
        predictions will serve to reduce noise in the data. Removing samples that produce
        correct predictions will serve to reduce redundant data 

        Parameters
        ----------

        Returns
        -------
        
        """
        # List to keep track of samples that should be dropped from the training data
        samples_to_drop = []

        for index, instance in self.training_data.iterrows():
            # Retrieve the relevant features for the given instance
            sample_vector = instance[self.features]

            # Predict the response value for the given sample
            prediction = self._classify(sample_vector, k)

            if [instance[self.target_feature]] != prediction:
                samples_to_drop.append(index)

        # Drop irrelevant samples from the training data
        self.training_data.drop(index=samples_to_drop, inplace=True)

    def _check_performance(self) -> float:
        """
        """
        # TODO: Update implementation of CV to reflect latest version of CV class 
        cv = CrossValidation(self.training_data, self.target_feature, regression=self.regression)
        results = cv.validate(KNN, 10, predict_params=[2])

        # Keep track of the performance of the model
        performance_metric = 0

        # Compute the performance metric for each fold from CV
        for result in results:
            if self.regression:
                performance_metric += EvaluationMeasure.calculate_means_square_error(result)
            else:
                performance_metric += EvaluationMeasure.calculate_0_1_loss(result)

        # Return the average performance of the model trained with the current 
        # training data
        return performance_metric / len(results)

    def predict(self, instance, k) -> str:
        # Measure performance of model on initial dataset
        initial_performance = self._check_performance()

        # Continue to minimize the training data if the performance does not decrease
        while(self._check_performance() >= initial_performance):
            self._minimize_data(k)

        # Report the prediction based on the minimized dataset
        return self._classify(instance, k)
            