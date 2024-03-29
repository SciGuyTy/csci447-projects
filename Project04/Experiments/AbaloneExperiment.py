import datetime
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

abalone_save_location = "./Project04/ExperimentSaves/abalone.objects"


def regression_output_transformer(output_vector: np.array):
    return output_vector[0]


def abalone_experiment_pso(runPSOTuning, network_shape):

    with open(abalone_save_location, 'rb') as f:
        training_test_data, pp, tuning_data, ___, cv = pickle.load(f)

    np = {'shape': network_shape, 'output_transformer': regression_output_transformer, 'regression': True,
          'random_weight_range': (-0.5, 0.5)}

    def individual_eval_method(fold, network):
        return EvaluationMeasure.calculate_means_square_error(cv.calculate_results_for_fold(network, fold.sample(frac=0.2)))

    hp = {'inertia': [0.1, 0.05, 0.3, 0.1], 'c1': [1.4, 1, 2, 0.3], 'c2': [1.4, 1, 2, 0.3]}
    hp_order = ['inertia', 'c1', 'c2']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if runPSOTuning:
        tu = TuningUtility(PSO, training_test_data, tuning_data, individual_eval_method, np, 15, 50, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'inertia': 0.25, 'c1': 1.6, 'c2': 1.6}

    population_size = 15
    generations = 50
    
    def individual_eval_method(fold, network):
        return EvaluationMeasure.calculate_means_square_error(cv.calculate_results_for_fold(network, fold))

    tu = TuningUtility(PSO, training_test_data, tuning_data, individual_eval_method, np, population_size, generations,
                       hp, hp_order)

    jobs = []
    manager = multiprocessing.Manager()
    fold_networks = manager.dict()
    for i, (training_data, hold_out, norm_params) in enumerate(training_test_data):
        process = multiprocessing.Process(target=tu.train_on_fold, args=(best_hp, training_data, fold_networks, i))
        jobs.append(process)
        process.start()

    for j in jobs:
        j.join()

    fold_results = [None] * len(training_test_data)
    for id, (network, fitness) in fold_networks.items():
        #print(Utilities.serialize_network(network))
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_data[id][1])

    mse = [EvaluationMeasure.calculate_means_square_error(i) for i in fold_results]
    print("MSE: ", mse)

def abalone_experiment_ga(run_tuning, network_shape):

    with open(abalone_save_location, 'rb') as f:
        training_test_data, pp, tuning_data, ___, cv = pickle.load(f)

    network_params = {'shape': network_shape, 'output_transformer': regression_output_transformer, 'regression': True,
          'random_weight_range': (-0.1, 0.1)}

    def individual_eval_method(fold, network):
        return EvaluationMeasure.calculate_means_square_error(cv.calculate_results_for_fold(network, fold.sample(frac=0.2)))


    hp = {'num_replaced_couples': [4, 1, 10, 3], 'tournament_size': [3, 2, 6, 2], 'probability_of_cross': [0.5, 0.1, 0.7, 0.2], 'probability_of_mutation': [0.05, 0.05, 0.25, 0.05], 'mutation_range': [0.1, 0.1, 0.5, 0.2], 'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation}
    hp_order = ['num_replaced_couples', 'tournament_size', 'probability_of_cross', 'probability_of_mutation', 'probability_of_mutation', 'mutation_range']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(Genetic, training_test_data, tuning_data, individual_eval_method, network_params, 15, 50, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation, 'num_replaced_couples': 2, 'tournament_size': 2, 'probability_of_cross': 0.5, 'probability_of_mutation': 0.05, 'mutation_range': 1}

    population_size = 15
    generations = 50
    
    def individual_eval_method(fold, network):
        return EvaluationMeasure.calculate_means_square_error(cv.calculate_results_for_fold(network, fold))
    tu = TuningUtility(Genetic, training_test_data, tuning_data, individual_eval_method, network_params, population_size, generations,
                       hp, hp_order)

    jobs = []
    manager = multiprocessing.Manager()
    fold_networks = manager.dict()
    for i, (training_data, hold_out, norm_params) in enumerate(training_test_data):
        process = multiprocessing.Process(target=tu.train_on_fold, args=(best_hp, training_data, fold_networks, i))
        jobs.append(process)
        process.start()

    for j in jobs:
        j.join()

    fold_results = [None] * len(training_test_data)
    for id, (network, fitness) in fold_networks.items():
        #print(Utilities.serialize_network(network))
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_data[id][1])

    mse = [EvaluationMeasure.calculate_means_square_error(i) for i in fold_results]
    print("MSE: ", mse)

def abalone_experiment_de(run_tuning, network_shape):

    with open(abalone_save_location, 'rb') as f:
        training_test_data, pp, tuning_data, ___, cv = pickle.load(f)

    network_params = {'shape': network_shape, 'output_transformer': regression_output_transformer, 'regression': True,
          'random_weight_range': (-0.1, 0.1)}

    def individual_eval_method(fold, network):
        return EvaluationMeasure.calculate_means_square_error(cv.calculate_results_for_fold(network, fold.sample(frac=.2)))

    hp = {'num_replaced_parents': [1, 1, 5, 1], 'mutation_scale_factor': 0.5, 'crossover_rate': [0.2, 0.1, 0.3, 0.05], 'crossover': BinomialCrossover}
    hp_order = ['num_replaced_parents', 'crossover_rate']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(DifferentialEvolution, training_test_data, tuning_data, individual_eval_method, network_params, 15, 50, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'num_replaced_parents': 3, 'mutation_scale_factor': 0.5, 'crossover_rate': 0.2, 'crossover': BinomialCrossover}
    population_size = 15
    generations = 50
    
    def individual_eval_method(fold, network):
        return EvaluationMeasure.calculate_means_square_error(cv.calculate_results_for_fold(network, fold))

    tu = TuningUtility(DifferentialEvolution, training_test_data, tuning_data, individual_eval_method, network_params, population_size, generations,
                       hp, hp_order)

    jobs = []
    manager = multiprocessing.Manager()
    fold_networks = manager.dict()
    for i, (training_data, hold_out, norm_params) in enumerate(training_test_data):
        process = multiprocessing.Process(target=tu.train_on_fold, args=(best_hp, training_data, fold_networks, i))
        jobs.append(process)
        process.start()

    for j in jobs:
        j.join()

    fold_results = [None] * len(training_test_data)
    for id, (network, fitness) in fold_networks.items():
        #print(training_test_folds[id][1])
        fold_results[id] = cv.calculate_results_for_fold(network, training_test_data[id][1])

    mse = [EvaluationMeasure.calculate_means_square_error(i) for i in fold_results]
    print("MSE: ", mse)
    
if __name__ == "__main__":
    print(datetime.datetime.now())
    # print("Starting ablone tuning ga 0")
    # abalone_experiment_ga(False, [8,1])
    # # MSE: [97.8363536423652, 103.74668215221638, 101.00912978247679, 90.0019449086871, 102.00958440543732,
    # #       111.66806180063219, 103.46568998490451, 110.29012221253555, 98.11118319067677, 91.66553966565934]
    # print("Starting ablone tuning ga 1")
    # 
    # abalone_experiment_ga(False, [8,8,1])
    # # MSE: [97.8363536423652, 103.74668215221638, 101.00912978247679, 90.0019449086871, 102.00958440543732,
    # #       111.66806180063219, 103.46568998490451, 110.29012221253555, 98.11118319067677, 91.66553966565934]
    # print("Starting ablone tuning ga 2")
    # 
    # abalone_experiment_ga(False, [8,8,8,1])
    # #MSE:  [97.0083592104536, 87.76732377933072, 93.82328066985096, 94.42286487239113, 88.24536272942275, 104.1578407995285, 99.76684413738005, 105.40079802670999, 92.6852413204904, 98.07797209960378]
    # 
    
    print("running abalone 0")
    abalone_experiment_de(False, [8,1])    


    print("running abalone 1")
    abalone_experiment_de(False, [8, 8, 1])
    

    print("running abalone 2")
    abalone_experiment_de(False, [8,8,8,1])