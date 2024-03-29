import multiprocessing
import pickle

import numpy as np

from Project04.Algorithms.Crossover.BinomialCrossover import BinomialCrossover
from Project04.Algorithms.Crossover.UniformCrossover import UniformCrossover
from Project04.Algorithms.DifferentialEvolution import DifferentialEvolution
from Project04.Algorithms.Genetic import Genetic
from Project04.Algorithms.Mutation.UniformMutation import UniformMutation
from Project04.Algorithms.PSO import PSO
from Project04.Algorithms.Selection.TournamentSelect import TournamentSelect
from Project04.Evaluation.CrossValidation import CrossValidation
from Project04.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project04.Utilities.Preprocess import Preprocessor
from Project04.Utilities.TuningUtility import TuningUtility
from Project04.Utilities.Utilities import Utilities


def output_transformer(output_vector: np.array):
    return output_vector.argmax() + 1

glass_save_location = "./Project04/ExperimentSaves/glass.objects"


def glass_experiment_pso(run_tuning, network_shape):

    with open(glass_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    np = {'shape': network_shape, 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-1, 1)}
    
    cv.classes = [i for i in range(1,8)]

    def individual_eval_method(fold, network):
        return 1-EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    hp = {'inertia': [0.2, 0.1, 0.4, 0.1], 'c1': [1.4, 1, 2, 0.2], 'c2': [1.4, 1, 2, 0.2]}
    hp_order = ['inertia', 'c1', 'c2']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(PSO, training_test_folds, tuning_data, individual_eval_method, np, 30, 50, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'inertia': 0.2, 'c1': 1.4, 'c2': 1.0}


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
    training_results = [None] * len(training_test_folds)
    for id, (network, fitness) in fold_networks.items():
        training_results[id] = 1-fitness
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_folds[id][1])

    loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in fold_results]
    f1 = [EvaluationMeasure.calculate_f_beta_score(i, 2) for i in fold_results]
    print("Loss: ", loss)
    print("F1", f1)
    print("Training Loss", training_results)

def glass_experiment_ga(run_tuning, network_shape):

    with open(glass_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)
    cv.classes = [i for i in range(1,8)]

    network_params = {'shape': network_shape, 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-0.1, 0.1)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))


    hp = {'num_replaced_couples': [1, 1, 6, 1], 'tournament_size': [2, 2, 5, 1], 'probability_of_cross': 0.5, 'probability_of_mutation': 0.05, 'mutation_range': [.3, 0.1, 1, 0.2], 'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation}
    hp_order = ['num_replaced_couples', 'tournament_size', 'mutation_range']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(Genetic, training_test_folds, tuning_data, individual_eval_method, network_params, 30, 100, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation, 'num_replaced_couples': 2, 'tournament_size': 2, 'probability_of_cross': 0.5, 'probability_of_mutation': 0.05, 'mutation_range': 0.1}


    population_size = 30
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
    tuning_results = [None] * len(training_test_folds)
    for id, (network, fitness) in fold_networks.items():
        tuning_results[id] = fitness
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_folds[id][1])

    tuning_results = [1-i for i in tuning_results]
    loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in fold_results]
    f1 = [EvaluationMeasure.calculate_f_beta_score(i, 2) for i in fold_results]
    print("Tuning Loss", tuning_results)
    print("Loss: ", loss)
    print("F1", f1)

def glass_experiment_de(run_tuning, network_shape):

    with open(glass_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)
    cv.classes = [i for i in range(1,8)]

    network_params = {'shape': network_shape, 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-0.7, 0.7)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    hp = {'num_replaced_parents': 4, 'mutation_scale_factor': [0.1, 0.1, 2.5, 0.5], 'crossover_rate': 0.5, 'crossover': BinomialCrossover}
    hp_order = ['mutation_scale_factor']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(DifferentialEvolution, training_test_folds, tuning_data, individual_eval_method, network_params, 30, 100, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'num_replaced_parents': 4, 'mutation_scale_factor': 1.6, 'crossover_rate': 0.5, 'crossover': BinomialCrossover}
    population_size = 30
    generations = 100
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
    training_results = [None] * len(training_test_folds)
    for id, (network, fitness) in fold_networks.items():
        training_results[id] = fitness
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_folds[id][1])

    loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in fold_results]
    f1 = [EvaluationMeasure.calculate_f_beta_score(i, 2) for i in fold_results]
    print("Loss: ", loss)
    print("F1", f1)

if __name__ == "__main__":

    #print("Running 0 glass pso")
    #glass_experiment_pso(False, [9, 7])
    #print("Running 2 glass pso")
    #glass_experiment_pso(False, [9,9,9, 7])
    # 1 layer pso
    # Loss: [0.13636363636363635, 0.47619047619047616, 0.47619047619047616, 0.35, 0.35, 0.3684210526315789,
    #        0.3888888888888889, 0.3888888888888889, 0.375, 0.1875]
    # F1[
    #     0.19047619047619047, 0.2, 0.0, 0.5185185185185185, 0.5185185185185185, 0.2, 0.56, 0.5454545454545455, 0.5454545454545454, 0.0]
    # Training
    # Loss[
    #     0.4497041420118344, 0.48235294117647065, 0.4529411764705882, 0.35672514619883033, 0.35672514619883033, 0.5116279069767442, 0.35260115606936415, 0.47976878612716756, 0.3542857142857143, 0.3771428571428572]

    #print("Tuning glass ga")
    # glass_experiment_ga(False, [9, 9, 7])
       # Loss: [0.13636363636363635, 0.3333333333333333, 0.3333333333333333, 0.3, 0.15, 0.3684210526315789,
    #        0.3888888888888889, 0.3888888888888889, 0.375, 0.375]
    # F1[0.0, 0.5, 0.0, 0.0, 0.0, 0.5384615384615384, 0.56, 0.56, 0.0, 0.0]

    # print("Running 0 glass ga")
    # glass_experiment_ga(False, [9,7])
    # print("Running 2 glass ga")
    # glass_experiment_ga(False, [9,9,9,7])
    # 
    # #print("Tuning glass de")
    # #glass_experiment_de(True, [9, 9, 7])
    
    print("Running 0 glass de")
    glass_experiment_de(False, [9,7])
    print("DE 0 ^^")

    print("Running 1 glass de")
    glass_experiment_de(False, [9, 9, 7])
    print("DE 1 ^^")

    print("Running 2 glass de")
    glass_experiment_de(False, [9,9,9, 7])
    print("DE 2 ^^")
