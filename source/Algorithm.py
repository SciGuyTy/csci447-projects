from typing import Any, List, Tuple
import pandas as pd

# TODO: Define types based on implementation
Classification = Any
Attribute = Tuple(str, Any)
Observation = dict[str, Any]

class Algorithm():
    training_data: pd.DataFrame = None
    training_classes: List[Any] = None
    training_attributes: List[str] = None

    def __init__(self, training_data: pd.DataFrame, classification_column: pd.Series) -> None:
        """Initialize the algorithm by decomposing training classifications and and attributes 
        from the training data

        Parameters
        ----------
        training_data: pandas.DataFrame
            A Pandas DataFrame object containing the training dataset.

        classification_column: pandas.Series
            A series that represents the column within the 
            DataFrame which contains classification data.
        """

        # Reference to the actual training data
        self.training_data = training_data

        # An array of possible classifications within the data
        self.training_classes: List[Any] = classification_column.unique
        
        # An array of attributes representing the objects of interest
        self.training_attributes = training_data.columns.drop(classification_column.name)
        
        # Setup any other instance variables

    # Define any utility methods

    def __get_classification_proportion(self, classification: Classification) -> float:
        """Compute the proportion of observations with the provided classification 
        relative to the total number of observations in the dataset
        
        Parameters
        ----------
        classification: Classification
            The classification with which to compute the relative proportion of observations.

        Returns
        -------
        proportion: float
            The proportion of observations with the provided classification 
            relative to the total number of observations in the dataset.
        """
        pass
    
    def __get_attribute_score(self, classification: Classification, attribute: Attribute) -> float:
        """Compute the proportion of observations within a classification
        that have a value equal to the novel observation for a given attribute

        Parameters
        ----------
        classification: Classification
            The classification with which the observations belong to.
        
        attribute: Attribute
            The attribute with which to compute the proportion of observations whose 
            values are equal to that of the novel observation.

        Returns
        -------
        proportion: float
            The proportion of observations within a classification that have a 
            value equal to the novel observation for a given attribute.
        """
        pass

    def predict(self, observation: Observation) -> Classification:
        """Classify a novel observation based on the observation's attributes
        
        observation: Observation 
            A novel observation with which to classify based on attribute values.]
            The attribute values should be stored as a dictionary with keys that match
            column names for the dataset.

        Returns
        -------
        prediction: Classification
            The predicted classification for the provided observation.
        """
        pass

