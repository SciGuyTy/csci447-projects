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

def breast_cancer_experiment_pso(run_tuning, network_shape):

    with open(breast_cancer_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    np = {'shape': network_shape, 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-1, 1)}

    def individual_eval_method(fold, network):
        return 1-EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    hp = {'inertia': [0.2, 0.1, 0.4, 0.1], 'c1': [1.4, 1, 2, 0.2], 'c2': [1.0, 0.5, 1.1, 0.1]}
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

def breast_cancer_experiment_ga(run_tuning, network_shape):

    with open(breast_cancer_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    network_params = {'shape': network_shape, 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-0.1, 0.1)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))


    hp = {'num_replaced_couples': 2, 'tournament_size': 4, 'probability_of_cross': 0.5, 'probability_of_mutation': 0.05, 'mutation_range': .3, 'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation}
    hp_order = ['num_replaced_couples']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(Genetic, training_test_folds, tuning_data, individual_eval_method, network_params, 30, 100, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation, 'num_replaced_couples': 2, 'tournament_size': 4, 'probability_of_cross': 0.5, 'probability_of_mutation': 0.05, 'mutation_range': 0.3}

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

def breast_cancer_experiment_de(run_tuning, network_shape):

    with open(breast_cancer_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    network_params = {'shape': network_shape, 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-0.7, 0.7)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    hp = {'num_replaced_parents': 1, 'mutation_scale_factor': [0.1, 0.1, 2.5, 0.5], 'crossover_rate': [0.2, 0.1, 0.3, 0.05], 'crossover': BinomialCrossover}
    hp_order = ['mutation_scale_factor', 'crossover_rate']
    # {'inertia': 0.1, 'c1': 1.4, 'c2': 0.6}

    if run_tuning:
        tu = TuningUtility(DifferentialEvolution, training_test_folds, tuning_data, individual_eval_method, network_params, 30, 100, hp, hp_order)
        best_hp = tu.tune_hyperparameters()
        print(best_hp)
    else:
        best_hp = {'num_replaced_parents': 1, 'mutation_scale_factor': 0.1, 'crossover_rate': 0.2, 'crossover': BinomialCrossover}
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
    breast_cancer_save_location = "../ExperimentSaves/breast_cancer.objects"
    breast_cancer_experiment_ga(False, [9, 2])
    # Tuning
    # Loss[
    #     0.6340579710144928, 0.8115942028985508, 0.822463768115942, 0.657608695652174, 0.6467391304347826, 0.6401446654611211, 0.6564195298372514, 0.6853526220614828, 0.7631103074141049, 0.7906137184115524]
    # Loss: [0.6290322580645161, 0.7419354838709677, 0.8225806451612904, 0.6451612903225806, 0.6451612903225806,
    #        0.6557377049180327, 0.7213114754098361, 0.7049180327868853, 0.7377049180327869, 0.7333333333333333]
    # F1[0.0, 0.5789473684210527, 0.8, 0.5, 0.0, 0.0, 0.32, 0.3076923076923077, 0.4666666666666666, 0.6190476190476191]
    
    breast_cancer_experiment_ga(False, [9, 2,2])
    # Tuning
    # Loss[
    #     0.6503623188405797, 0.6503623188405797, 0.3496376811594203, 0.6503623188405797, 0.6503623188405797, 0.6491862567811935, 0.6491862567811935, 0.6491862567811935, 0.6491862567811935, 0.6498194945848376]
    # Loss: [0.6451612903225806, 0.6451612903225806, 0.3548387096774194, 0.6451612903225806, 0.6451612903225806,
    #        0.6557377049180327, 0.6557377049180327, 0.6557377049180327, 0.6557377049180327, 0.65]
    # F1[0.0, 0.0, 0.5238095238095238, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    breast_cancer_experiment_ga(False, [9, 2,2,2])
    # Tuning
    # Loss[
    #     0.6503623188405797, 0.6503623188405797, 0.6503623188405797, 0.6503623188405797, 0.3496376811594203, 0.6491862567811935, 0.6491862567811935, 0.6491862567811935, 0.6491862567811935, 0.6498194945848376]
    # Loss: [0.6451612903225806, 0.6451612903225806, 0.6451612903225806, 0.6451612903225806, 0.3548387096774194,
    #        0.6557377049180327, 0.6557377049180327, 0.6557377049180327, 0.6557377049180327, 0.65]
    # F1[0.0, 0.0, 0.0, 0.0, 0.5238095238095238, 0.0, 0.0, 0.0, 0.0, 0.0]

    
    
    #Loss:  [0.43548387096774194, 0.6290322580645161, 0.5483870967741935, 0.5806451612903226, 0.5483870967741935, 0.6065573770491803, 0.819672131147541, 0.8688524590163934, 0.6885245901639344, 0.8]
    #F1 [0.5569620253164557, 0.0, 0.6111111111111112, 0.6060606060606061, 0.6111111111111112, 0.5555555555555556, 0.7843137254901961, 0.8260869565217391, 0.6885245901639345, 0.75]
    #breast_cancer_experiment_de(False, [9, 9, 2])
    #Loss: [0.6451612903225806, 0.6451612903225806, 0.3548387096774194, 0.3548387096774194, 0.3548387096774194,  0.3442622950819672, 0.6557377049180327, 0.3442622950819672, 0.3442622950819672, 0.65]
    #F1[        0.0, 0.0, 0.5238095238095238, 0.5238095238095238, 0.5238095238095238, 0.5121951219512195, 0.0, 0.5121951219512195, 0.5121951219512195, 0.0]
    #breast_cancer_experiment_de(False, [9,9, 9, 2])
    # Loss: [0.6451612903225806, 0.3548387096774194, 0.3548387096774194, 0.6451612903225806, 0.3548387096774194,
    #        0.3442622950819672, 0.6557377049180327, 0.6557377049180327, 0.6557377049180327, 0.65]
    # F1[0.0, 0.5238095238095238, 0.5238095238095238, 0.0, 0.5238095238095238, 0.5121951219512195, 0.0, 0.0, 0.0, 0.0]