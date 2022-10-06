from source.Evaluation.CrossValidation import CrossValidation
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Algorithms.DistanceFunctions.DistanceFunction import DistanceFunction
from source.Algorithms.KNN import KNN
import pandas as pd

class EditedKNN():
    def __init__(
        self, 
        target_feature: str,
        data: pd.DataFrame,
        regression: bool = False,
        distance_function: DistanceFunction = Minkowski(),
    ):
        self.target_feature = target_feature
        self.data = data
        self.regression = regression
        self.distance_function = distance_function

        self.features = self.data.columns.drop(target_feature)

        self.knn = KNN(data, target_feature, regression, distance_function)

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
        samples_to_drop = []

        for index, instance in self.data.iterrows():
            sample_vector = instance[self.features]

            prediction = self.knn.predict(sample_vector, k)

            if [instance[self.target_feature]] != prediction:
                samples_to_drop.append(index)

        self.data.drop(index=samples_to_drop, inplace=True)

    def _check_performance(self) -> float:
        cv = CrossValidation(self.data, "class")
        results = cv.validate(KNN, 10, self.regression, predict_params=[2])

        perf = 0

        for result in results:
            num_correct_predictions = 0

            # Compute the total number of correct predictions (TP for each class level)
            for column in result.columns:
                num_correct_predictions += result[column][column]

            perf += num_correct_predictions / result.to_numpy().sum()

        return perf / len(results)

    def predict(self, instance, k) -> str:
        current_performance = self._check_performance()

        while(self._check_performance() >= current_performance):
            print(current_performance)
            self._minimize_data(k)

        return self.knn.predict(instance, k)
            