import multiprocessing
import pickle
from datetime import time, datetime

import numpy as np

import pandas as pd

from Project04.Algorithms.Crossover.BinomialCrossover import BinomialCrossover
from Project04.Algorithms.Crossover.UniformCrossover import UniformCrossover
from Project04.Algorithms.Genetic import Genetic
from Project04.Algorithms.Mutation.UniformMutation import UniformMutation
from Project04.Algorithms.Selection.TournamentSelect import TournamentSelect
from Project04.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project04.Utilities.TuningUtility import TuningUtility
from Project04.Algorithms.PSO import PSO
from Project04.Evaluation.CrossValidation import CrossValidation
from Project04.NeuralNetwork import NeuralNetwork
from Project04.Evaluation.EvaluationCallable import EvaluationCallable
from Project04.Utilities.Preprocess import Preprocessor
from Project04.Utilities.Utilities import Utilities
from Project04.Experiments import SoybeanExperiment, BreastCancerExperiment

def output_transformer(output_vector: np.array):
    return output_vector.argmax() + 1

def test_generate_evals():

    def individual_eval_method(x, y):
        return x * y

    folds = [i for i in range(5)]

    methods = EvaluationCallable.generate_eval_methods_for_folds(folds, individual_eval_method)

    for method in methods:
        print(method(2))

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

def test_GA():

    networks = []

    num_of_gen = 1000
    population_size = 10

    for i in range(population_size):
        networks.append(NeuralNetwork([2,1], lambda x : x, True, (0, 1)))

    hp = {'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation,
               'num_replaced_couples': 1, 'tournament_size': 4, 'probability_of_cross': 0.5,
               'probability_of_mutation': 0.05, 'mutation_range': 0.1}

    def evaluation_method(fold, network):
        network = Utilities.serialize_network(network)
        return np.sum(np.abs(network))

    method = EvaluationCallable(2, evaluation_method)

    ga = Genetic(networks, hp, method)
    
    print(ga.train(num_of_gen))


def test_DE():
    networks = []

    num_of_gen = 50
    population_size = 4

    for i in range(population_size):
        networks.append(NeuralNetwork([2, 1], lambda x: x, True, (0, 1)))

    hp = {'num_replaced_parents': 1, 'mutation_scale_factor': 0.1, 'crossover_rate': 0.2, 'crossover': BinomialCrossover}

    def evaluation_method(fold, network):
        network = Utilities.serialize_network(network)
        return np.sum(np.abs(network))

    method = EvaluationCallable(2, evaluation_method)

    de = Genetic(networks, hp, method)

    print(de.train(num_of_gen))



if __name__ == "__main__":
    #SoybeanExperiment.soybean_experiment_pso(False)
    # SoybeanExperiment.soybean_experiment_ga(False)
    # SoybeanExperiment.soybean_experiment_de(False)
    
    #BreastCancerExperiment.breast_cancer_experiment_pso(False)
    # print(datetime.now())
    # BreastCancerExperiment.breast_cancer_experiment_de(True, [9,9,2])
    # print(datetime.now())
    test_GA()


    pass
