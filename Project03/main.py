import math

import numpy as np
import pandas as pd

from Project03.ActivationFunctions.Sigmoid import Sigmoid
from Project03.ActivationFunctions.Softmax import Softmax
from Project03.Evaluation.CrossValidation import CrossValidation
from Project03.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project03.Layer import Layer
from Project03.NeuralNetwork import NeuralNetwork
from Project03.Utilities.Preprocess import Preprocessor
from Project03.Utilities.TuningUtility import TuningUtility


def output_transformer(output_vector: np.array):
    return output_vector.argmax() + 1

def regression_output_transformer(output_vector: np.array):
    return output_vector[0]

def output_transformer_test(output_vector: np.array):
    return 1 - output_vector.argmax()
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
    cv.data = cv.data.drop(tuning_data.index)

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
        "epochs": 100,
    }
    tu = TuningUtility(training_test_folds, tuning_data, "class", 2, 2, classification_modifier, output_transformer_test, True, training_params)
    best_models = tu.tune_for_h_hidden_layers(2)
    print(best_models)

    overall_results = cv.validate_for_folds(training_test_folds, best_models)
    print(overall_results)

def glass_identification_experiment():
    file_path = "../datasets/classification/GlassIdentification/glass.data"

    column_names = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]

    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(file_path, column_names,  dropNA=['?'], columns_to_drop=['ID'])
    cv = CrossValidation(PP.data, "class", False)
    tuning_data = cv.get_tuning_set(0.1)
    cv.data = cv.data.drop(tuning_data.index)

    folds = cv.fold_data(10, True)
    training_test_folds = cv.get_training_test_data_from_folds(folds)

    classes = PP.data["class"].unique()

    def classification_modifier(pattern: pd.Series):
        target = [0] * 7
        target[pattern.loc["class"] - 1] = 1
        return target

    training_params = {
        "learning_rate": .01,
        "momentum": 0.005,
        "batch_size": 10,
        "epochs": 200,
    }
    tu = TuningUtility(training_test_folds, tuning_data, "class", 9, 7, classification_modifier, output_transformer,
                       True, training_params)
    best_models = tu.tune_for_h_hidden_layers(0)
    print(best_models)

    overall_results = cv.validate_for_folds(training_test_folds, best_models)
    print(overall_results)


def breast_cancer_experiment():
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
    cv.data = cv.data.drop(tuning_data.index)

    folds = cv.fold_data(10, True)
    training_test_folds = cv.get_training_test_data_from_folds(folds)

    classes = PP.data["class"].unique()

    def classification_modifier(pattern: pd.Series):
        target = [0] * len(classes)
        target[pattern.loc["class"] - 1] = 1
        return target

    training_params = {
        "learning_rate": .001,
        "momentum": 0,
        "batch_size": 10,
        "epochs": 200,
    }
    tu = TuningUtility(training_test_folds, tuning_data, "class", 9, 2, classification_modifier, output_transformer, True, training_params)
    best_models = tu.tune_for_h_hidden_layers(1)
    print(best_models)

    overall_results = cv.validate_for_folds(training_test_folds, best_models)
    print(overall_results)

def soybean_experiment():
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
    cv.data = cv.data.drop(tuning_data.index)

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
        "epochs": 10,
    }
    tu = TuningUtility(training_test_folds, tuning_data, "class", 35, 4, classification_modifier, output_transformer, True, training_params)
    best_models = tu.tune_for_h_hidden_layers(1)
    print(best_models)


def abalone_experiment():
    pp = Preprocessor()

    file_path = '../datasets/regression/Abalone/abalone.data'
    column_names = [
        'sex',
        'length',
        'diameter',
        'height',
        'whole_weight',
        'shucked_weight',
        'viscera_weight',
        'shell_weight',
        'rings',
    ]
    sex_map = {"M": 1, "F": 2, "I": 3}

    feature_modifiers = {"sex": lambda x: sex_map[x]}

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        converters=feature_modifiers
    )
    # pp.data = pp.data[pp.data.area != 0]
    cv = CrossValidation(pp.data, "rings", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "rings", regression=True)

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    def regression_modifier(pattern: pd.Series):
        return [pattern['rings']]

    training_params = {
        "learning_rate": 0.001,
        "momentum": 0.01,
        "batch_size": 5,
        "epochs": 100,
    }
    tu = TuningUtility(training_test_data, tuning_data, "rings", 8, 1, regression_modifier,
                       regression_output_transformer, False, training_params)
    best_models = tu.tune_for_h_hidden_layers(1)
    print(best_models)

    overall_results = cv.validate_for_folds(training_test_data, best_models)
    print(overall_results)
    mse = [EvaluationMeasure.calculate_means_square_error(i) for i in overall_results]
    print(mse)

def forest_fire_experiment():
    pp = Preprocessor()

    file_path = "../datasets/regression/ForestFires/forestfires.csv"
    column_names = [
        "x",
        "y",
        "month",
        "day",
        "ffmc",
        "dmc",
        "dc",
        "isi",
        "temp",
        "rh",
        "wind",
        "rain",
        "area",
        ]
    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    day_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}

    feature_modifiers = {"month": lambda x: (month_map[x]), "day": lambda x: day_map[x], "area": lambda x: math.log(float(x)+1)}

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        converters=feature_modifiers
    )
    #pp.data = pp.data[pp.data.area != 0]
    cv = CrossValidation(pp.data, "area", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "area", regression=True)

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    def regression_modifier(pattern: pd.Series):
        return [pattern['area']]

    training_params = {
        "learning_rate": 0.01,
        "momentum": 0.0,
        "batch_size": 10,
        "epochs": 100,
    }
    tu = TuningUtility(training_test_data, tuning_data, "area", 12, 1, regression_modifier, regression_output_transformer, False, training_params)
    best_models = tu.tune_for_h_hidden_layers(1)
    print(best_models)

    overall_results = cv.validate_for_folds(training_test_data, best_models)
    print(overall_results)
    mse = [EvaluationMeasure.calculate_means_square_error(i) for i in overall_results]
    print(mse)


def single_forest_fire_experiment():
    pp = Preprocessor()

    file_path = "../datasets/regression/ForestFires/forestfires.csv"
    column_names = [
        "x",
        "y",
        "month",
        "day",
        "ffmc",
        "dmc",
        "dc",
        "isi",
        "temp",
        "rh",
        "wind",
        "rain",
        "area",
    ]
    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
                 "nov": 11, "dec": 12}
    day_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}

    feature_modifiers = {"month": lambda x: (month_map[x]), "day": lambda x: day_map[x],
                         "area": lambda x: math.log(float(x) + 1)
                         }

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        converters=feature_modifiers
    )
    # pp.data = pp.data[pp.data.area != 0]
    cv = CrossValidation(pp.data, "area", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "area", regression=True)

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    def regression_modifier(pattern: pd.Series):
        return [pattern['area']]

    training_params = {
        "learning_rate": 0.01,
        "momentum": 0.00,
        "batch_size": 10,
        "epochs": 250,
    }
    tu = TuningUtility(training_test_data, tuning_data, "area", 12, 1, regression_modifier,
                       regression_output_transformer, False, training_params)
    results = dict()
    best_models = tu.tune_single_fold(1, [[12, 6, 3, 1]], training_test_data[0][0], tuning_data, training_test_data[0][2],
                                      results)
    print(results)

def single_abalone_experiment():
    pp = Preprocessor()

    file_path = '../datasets/regression/Abalone/abalone.data'
    column_names = [
        'sex',
        'length',
        'diameter',
        'height',
        'whole_weight',
        'shucked_weight',
        'viscera_weight',
        'shell_weight',
        'rings',
    ]
    sex_map = {"M": 1, "F": 2, "I": 3}

    feature_modifiers = {"sex": lambda x: sex_map[x]}

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        converters=feature_modifiers
    )
    # pp.data = pp.data[pp.data.area != 0]
    cv = CrossValidation(pp.data, "rings", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "rings", regression=True)

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    def regression_modifier(pattern: pd.Series):
        return [pattern['rings']]

    training_params = {
        "learning_rate": 0.001,
        "momentum": 0.01,
        "batch_size": 5,
        "epochs": 100,
    }
    tu = TuningUtility(training_test_data, tuning_data, "rings", 8, 1, regression_modifier,
                       regression_output_transformer, False, training_params)
    results = dict()
    best_models = tu.tune_single_fold(1, [[8,8,1]], training_test_data[0][0], tuning_data, training_test_data[0][2], results)
    print(results)

if __name__ == "__main__":
    test_experiment()