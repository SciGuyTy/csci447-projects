import multiprocessing
import pickle

import numpy as np

from Project04.Algorithms.DifferentialEvolution import DifferentialEvolution
from Project04.Algorithms.Crossover.BinomialCrossover import BinomialCrossover
from Project04.Algorithms.Crossover.UniformCrossover import UniformCrossover
from Project04.Algorithms.Mutation.UniformMutation import UniformMutation
from Project04.Algorithms.PSO import PSO
from Project04.Algorithms.Genetic import Genetic
from Project04.Algorithms.Selection.TournamentSelect import TournamentSelect
from Project04.Evaluation.CrossValidation import CrossValidation
from Project04.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project04.Utilities.Preprocess import Preprocessor
from Project04.Utilities.TuningUtility import TuningUtility
from Project04.Utilities.Utilities import Utilities

soybean_save_location = "./ExperimentSaves/soybean.objects"


def output_transformer(output_vector: np.array):
    return output_vector.argmax() + 1


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


def soybean_experiment_pso(runPSOTuning):

    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    np = {'shape': [35, 4], 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-0.5, 0.5)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    hp = {'inertia': [0.2, 0.05, 0.3, 0.05], 'c1': [1.6, 1, 2, 0.2], 'c2': [1.6, 1, 2, 0.2]}
    hp_order = ['inertia', 'c1', 'c2']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if runPSOTuning:
        tu = TuningUtility(PSO, training_test_folds, tuning_data, individual_eval_method, np, 30, 100, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'inertia': 0.25, 'c1': 1.7999999999999998, 'c2': 1.0}

    population_size = 100
    generations = 200
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
        #print(Utilities.serialize_network(network))
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_folds[id][1])

    loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in fold_results]
    f1 = [EvaluationMeasure.calculate_f_beta_score(i, 2) for i in fold_results]
    print("Loss: ", loss)
    print("F1", f1)

def soybean_experiment_ga(run_tuning):

    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    network_params = {'shape': [35, 4], 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-0.1, 0.1)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))


    hp = {'num_replaced_couples': [4, 1, 10, 2], 'tournament_size': [3, 2, 6, 1], 'probability_of_cross': [0.8, 0.1, 1, 0.2], 'probability_of_mutation': [0.15, 0.05, 0.25, 0.05], 'mutation_range': (-1, 1), 'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation}
    hp_order = ['num_replaced_couples', 'tournament_size', 'probability_of_cross', 'probability_of_mutation', 'probability_of_mutation']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(Genetic, training_test_folds, tuning_data, individual_eval_method, network_params, 100, 100, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation, 'num_replaced_couples': 4, 'tournament_size': 3, 'probability_of_cross': 0.8, 'probability_of_mutation': 0.15, 'mutation_range': (-1, 1)}

    population_size = 100
    generations = 100
    tu = TuningUtility(Genetic, training_test_folds, tuning_data, individual_eval_method, network_params, population_size, generations,
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
        #print(Utilities.serialize_network(network))
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_folds[id][1])

    loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in fold_results]
    f1 = [EvaluationMeasure.calculate_f_beta_score(i, 2) for i in fold_results]
    print("Loss: ", loss)
    print("F1", f1)

def soybean_experiment_de(run_tuning):

    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    network_params = {'shape': [35, 4], 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-0.1, 0.1)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    hp = {'num_replaced_parents': [1, 1, 4, 1], 'mutation_scale_factor': [1.6, 0.5, 2.5, 0.5], 'crossover_rate': [0.2, 0.1, 0.3, 0.05], 'crossover': BinomialCrossover}
    hp_order = ['num_replaced_parents', 'mutation_scale_factor', 'crossover_rate']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(DifferentialEvolution, training_test_folds, tuning_data, individual_eval_method, network_params, 100, 100, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'num_replaced_parents': 1, 'mutation_scale_factor': 1.5, 'crossover_rate': 0.2, 'crossover': BinomialCrossover}
    population_size = 10
    generations = 1
    tu = TuningUtility(DifferentialEvolution, training_test_folds, tuning_data, individual_eval_method, network_params, population_size, generations,
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
        #print(training_test_folds[id][1])
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_folds[id][1])

    loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in fold_results]
    f1 = [EvaluationMeasure.calculate_f_beta_score(i, 2) for i in fold_results]
    print("Loss: ", loss)
    print("F1", f1)