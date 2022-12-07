import multiprocessing
import pickle
from datetime import time, datetime

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






if __name__ == "__main__":
    #SoybeanExperiment.soybean_experiment_pso(False)
    # SoybeanExperiment.soybean_experiment_ga(False)
    # SoybeanExperiment.soybean_experiment_de(False)
    
    #BreastCancerExperiment.breast_cancer_experiment_pso(False)
    print(datetime.now())
    BreastCancerExperiment.breast_cancer_experiment_ga(True)
    print(datetime.now())


    pass
