import multiprocessing
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
from Project04.Utilities.Utilities import Utilities


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

    num_of_gen = 10
    population_size = 5

    for i in range(population_size):
        networks.append(NeuralNetwork([3,3], lambda x : x, True, (0, 1)))

    hp = {'inertia': 0.1, 'c1': 1.5, 'c2': 3}

    def evaluation_method(fold, network):
        network = Utilities.serialize_network(network)
        return abs(np.sum(network)) + fold

    method = EvaluationCallable(2, evaluation_method)

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
          'random_weight_range': (-1, 1)}

    def individual_eval_method(fold, network):
        return 1-EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    hp = {'inertia': [0.4, 0.0, 0.8, 0.2], 'c1': [1.6, 1, 2, 0.2], 'c2': [1.4, 1, 2, 0.2]}
    hp_order = ['inertia', 'c1', 'c2'][::-1]
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(PSO, training_test_folds, tuning_data, individual_eval_method, np, 30, 50, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'inertia': 0.4, 'c1': 1.6, 'c2': 1.4}


    population_size = 30
    generations = 100
    tu = TuningUtility(PSO, training_test_folds, tuning_data, individual_eval_method, np, population_size, generations, hp, hp_order)

    jobs = []
    manager = multiprocessing.Manager()
    fold_networks = manager.dict()
    for i, (training_data, hold_out, norm_params) in enumerate(training_test_folds):
        process = multiprocessing.Process(target=tu.train_on_fold, args=(best_hp, training_data, fold_networks, i))
        jobs.append(process)
        process.start()

    for j in jobs:
        j.join()

    fold_results = [None] * len(training_test_folds)
    for id, network in fold_networks.items():
        print(Utilities.serialize_network(network))
        fold_results[id] = cv.calculate_results_for_fold(network, hold_out)

    loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in fold_results]
    f1 = [EvaluationMeasure.calculate_f_beta_score(i, 2) for i in fold_results]
    print("Loss: ", loss)
    print("F1", f1)

soybean_save_location = "./ExperimentSaves/soybean.objects"
def initialize_soybean_experiment():
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

    with open(soybean_save_location, 'wb+') as f:
        pickle.dump([training_test_folds, PP, tuning_data, folds, training_test_folds, cv], f)

def soybean_experiment(runPSOTuning):

    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    np = {'shape': [35, 4], 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-1, 1)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    hp = {'inertia': [0.4, 0.0, 0.8, 0.2], 'c1': [1.6, 1, 2, 0.2], 'c2': [1.6, 1, 2, 0.2]}
    hp_order = ['inertia', 'c1', 'c2'][::-1]
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if runPSOTuning:
        tu = TuningUtility(PSO, training_test_folds, tuning_data, individual_eval_method, np, 30, 50, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'inertia': 0.2, 'c1': 1.5999999999999999, 'c2': 1.2}

    population_size = 30
    generations = 100
    tu = TuningUtility(PSO, training_test_folds, tuning_data, individual_eval_method, np, population_size, generations,
                       hp, hp_order)

    jobs = []
    manager = multiprocessing.Manager()
    fold_networks = manager.dict()
    for i, (training_data, hold_out, norm_params) in enumerate(training_test_folds):
        process = multiprocessing.Process(target=tu.train_on_fold, args=(best_hp, training_data, fold_networks, i))
        jobs.append(process)
        process.start()

    for j in jobs:
        j.join()

    fold_results = [None] * len(training_test_folds)
    for id, network in fold_networks.items():
        print(Utilities.serialize_network(network))
        fold_results[id] = cv.calculate_results_for_fold(network, hold_out)

    loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in fold_results]
    f1 = [EvaluationMeasure.calculate_f_beta_score(i, 2) for i in fold_results]
    print("Loss: ", loss)
    print("F1", f1)



if __name__ == "__main__":
    #test_PSO()
    soybean_experiment(True)

    pass