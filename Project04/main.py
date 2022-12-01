import pickle

import numpy as np

import pandas as pd

from Project04.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project04.Utilities.TuningUtility import TuningUtility
from Project04.Algorithms.PSO import PSO
from Project04.Evaluation.CrossValidation import CrossValidation
from Project04.NeuralNetwork import NeuralNetwork
from Project04.Evaluation.EvaluationCallable import EvaluationCallable
from Project04.Utilities.Preprocess import Preprocessor

def output_transformer(output_vector: np.array):
    return output_vector.argmax() + 1

def test_generate_evals():

    def individual_eval_method(x, y):
        return x * y

    folds = [i for i in range(5)]

    methods = EvaluationCallable.generate_eval_methods_for_folds(folds, individual_eval_method)

    for method in methods:
        print(method(2))

#test_generate_evals()

def test_PSO():

    networks = []

    num_of_gen = 1000
    population_size = 100

    for i in range(population_size):
        networks.append(NeuralNetwork([3,3], lambda x : x, True, (0, 1)))

    hp = {'inertia': 0.1, 'c1': 1.5, 'c2': 3}

    def evaluation_method(fold, network):
        return abs(np.sum(network)) + fold

    method = EvaluationCallable(1, evaluation_method)

    pso = PSO(networks, hp, method)

    print(pso.train(num_of_gen))

breast_cancer_save_location = "./ExperimentSaves/breast_cancer.objects"
def initialize_breast_cancer_experiment():
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

    with open(breast_cancer_save_location, 'wb+') as f:
        pickle.dump([training_test_folds, PP, tuning_data, folds, training_test_folds, cv], f)

def breast_cancer_experiment(run_tuning):

    with open(breast_cancer_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    np = {'shape': [9, 9, 2], 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-.1, 0.1)}

    def individual_eval_method(fold, network):
        return 1-EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    if run_tuning:
        hp = {'inertia': [0.1, 0.05, 0.15, 0.05], 'c1': [1.4, 1, 1.6, 0.2], 'c2': [0.8, .2, 1.0, 0.2]}
        hp_order = ['inertia', 'c1', 'c2']
        tu = TuningUtility(PSO, folds, tuning_data, individual_eval_method, np, 30, 100, hp, hp_order)
        # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    population_size = 100
    generations = 100

    results = []
    for fold in folds:
        network, fitness = tu.train_on_fold(best_hp, fold)



if __name__ == "__main__":
    breast_cancer_experiment(False)

    pass