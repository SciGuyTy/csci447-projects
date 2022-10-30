import numpy as np
import pandas as pd

from Project03.ActivationFunctions.Sigmoid import Sigmoid
from Project03.ActivationFunctions.Softmax import Softmax
from Project03.Evaluation.CrossValidation import CrossValidation
from Project03.Layer import Layer
from Project03.NeuralNetwork import NeuralNetwork
from Project03.Utilities.Preprocess import Preprocessor
from Project03.Utilities.TuningUtility import TuningUtility


def output_transformer(output_vector: np.array):
    return output_vector.argmax() + 1

def output_transformer_test(output_vector: np.array):
    return output_vector.argmax()
def test_experiment():
    file_path = "../datasets/classification/test-classification.data"

    column_names = [
        "a",
        "b",
        "class",
    ]


    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(file_path, column_names)
    cv = CrossValidation(PP.data, "class", False)
    tuning_data = cv.get_tuning_set(0.1)
    cv.data.drop(tuning_data.index)

    folds = cv.fold_data(10, True)
    training_test_folds = cv.get_training_test_data_from_folds(folds)

    classes = PP.data["class"].unique()

    def classification_modifier(pattern: pd.Series):
        target = [0] * len(classes)
        target[pattern.loc["class"] - 1] = 1
        return target

    training_params = {
        "learning_rate": .1,
        "momentum": 0,
        "batch_size": 1,
        "epochs": 50,
    }
    tu = TuningUtility(training_test_folds, tuning_data, "class", 2, 2, classification_modifier, output_transformer_test, True, training_params)
    best_models = tu.tune_for_h_hidden_layers(0)
    print(best_models)

    overall_results = cv.validate_for_folds(training_test_folds, best_models)
    print(overall_results)

def breast_cancer_expirement():
    file_path = "../datasets/classification/BreastCancer/breast-cancer-wisconsin.data"

    column_names = [
        "id",
        "clump",
        "size",
        "shape",
        "adhesion",
        "epithelial_size",
        "nuclei",
        "chromatin",
        "nucleoli",
        "mitoses",
        "class",
    ]

    # Define a converter that converts the class values into True or False
    converters = {"class": lambda x: 2 if int(x) == 4 else 1}

    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(file_path, column_names, converters=converters, dropNA=['?'], columns_to_drop=['id'])
    cv = CrossValidation(PP.data, "class", False)
    tuning_data = cv.get_tuning_set(0.1)
    cv.data.drop(tuning_data.index)

    folds = cv.fold_data(10, True)
    training_test_folds = cv.get_training_test_data_from_folds(folds)

    classes = PP.data["class"].unique()

    def classification_modifier(pattern: pd.Series):
        target = [0] * len(classes)
        target[pattern.loc["class"] - 1] = 1
        return target

    training_params = {
        "learning_rate": .01,
        "momentum": 0,
        "batch_size": 25,
        "epochs": 100,
    }
    tu = TuningUtility(training_test_folds, tuning_data, "class", 9, 2, classification_modifier, output_transformer, True, training_params)
    best_models = tu.tune_for_h_hidden_layers(0)
    print(best_models)

    overall_results = cv.validate_for_folds(training_test_folds, best_models)
    print(overall_results)

def soybean_expirement():
    file_path = "../datasets/classification/Soybean/soybean-small.data"
    column_names = [
        "date",
        "plant_stand",
        "precip",
        "temp",
        "hail",
        "crop_hist",
        "area_damaged",
        "severity",
        "seed_tmt",
        "germination",
        "plant_growth",
        "leaves",
        "leafspots_halo",
        "leafspots_marg",
        "leafspot_size",
        "leaf_shread",
        "leaf_malf",
        "leaf_mild",
        "stem",
        "lodging",
        "stem_cankers",
        "canker_lesion",
        "fruiting_bodies",
        "external_decay",
        "mycelium",
        "int_discolor",
        "sclerotia",
        "fruit_pods",
        "fruit_spots",
        "seed",
        "mold_growth",
        "seed_discolor",
        "seed_size",
        "shriveling",
        "roots",
        "class",
    ]

    converters = {
        "class": lambda x: int(x[1])
    }

    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(file_path, column_names, converters=converters)
    cv = CrossValidation(PP.data, "class", False)
    tuning_data = cv.get_tuning_set(0.1)
    cv.data.drop(tuning_data.index)

    folds = cv.fold_data(10, True)
    training_test_folds = cv.get_training_test_data_from_folds(folds)

    classes = PP.data["class"].unique()

    def classification_modifier(pattern: pd.Series):
        target = [0] * len(classes)
        target[pattern.loc["class"] - 1] = 1
        return target

    training_params = {
        "learning_rate": 0.01,
        "momentum": 0.1,
        "batch_size": 1,
        "epochs": 100,
    }
    tu = TuningUtility(training_test_folds, tuning_data, "class", 35, 4, classification_modifier, output_transformer, True, training_params)
    best_models = tu.tune_for_h_hidden_layers(0)
    print(best_models)


if __name__ == "__main__":
    test_experiment()