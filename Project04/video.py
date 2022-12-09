import math
import pickle
import random

import numpy as np
import pandas as pd

from Project04.Algorithms.Crossover.BinomialCrossover import BinomialCrossover
from Project04.Algorithms.Crossover.UniformCrossover import UniformCrossover
from Project04.Algorithms.DifferentialEvolution import DifferentialEvolution
from Project04.Algorithms.Genetic import Genetic
from Project04.Algorithms.Mutation.UniformMutation import UniformMutation
from Project04.Algorithms.PSO import Particle
from Project04.Algorithms.Selection.TournamentSelect import TournamentSelect
from Project04.Evaluation.EvaluationCallable import EvaluationCallable
from Project04.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project04.NeuralNetwork import NeuralNetwork
from Project04.Utilities.TuningUtility import TuningUtility
from Project04.Utilities.Utilities import Utilities

soybean_save_location = "./ExperimentSaves/soybean.objects"


def output_transformer(output_vector: np.array):
    return output_vector.argmax() + 1


def demonstrate_test_fold_output_classification():
    print("------ Demonstrate Output/Performance of Test Folds ------")

    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    print(f"Consider the following fold used for training the following algorithms: \n{training_test_folds[0][1]}".replace("\n", "\n\t\t"))
    print("")

    training_data = PP.data

    network_params = {'shape': [35, 4], 'output_transformer': output_transformer, 'regression': False,
          'random_weight_range': (-0.1, 0.1)}

    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    # ----- Genetic Algorithm -----
    hp = {'num_replaced_couples': [4, 1, 10, 2], 'tournament_size': [3, 2, 6, 1],
          'probability_of_cross': [0.8, 0.1, 1, 0.2], 'probability_of_mutation': [0.15, 0.05, 0.25, 0.05],
          'mutation_range': (-1, 1), 'selection': TournamentSelect, 'crossover': UniformCrossover,
          'mutation': UniformMutation}
    best_hp = {'selection': TournamentSelect, 'crossover': UniformCrossover, 'mutation': UniformMutation,
               'num_replaced_couples': 4, 'tournament_size': 3, 'probability_of_cross': 0.8,
               'probability_of_mutation': 0.15, 'mutation_range': 1}
    hp_order = ['num_replaced_couples', 'tournament_size', 'probability_of_cross', 'probability_of_mutation',
                'probability_of_mutation']

    population_size = 30
    generations = 2
    tu = TuningUtility(Genetic, training_test_folds, tuning_data, individual_eval_method, network_params,
                       population_size, generations,
                       hp, hp_order)

    network, fitness = tu.train_on_fold(best_hp, training_data)


    fold_results = cv.calculate_results_for_fold(network, training_test_folds[0][1])

    loss = EvaluationMeasure.calculate_0_1_loss(fold_results)
    f1 = EvaluationMeasure.calculate_f_beta_score(fold_results, 2)

    print(f"Genetic algorithm results on network with shape [35, 4], trained for 50 generations and a population of size 30")
    print(f"\tNetwork Weights:")
    for layer_num, layer in enumerate(network.layers):
        print(f"\t\tLayer {layer_num + 1}: \n{layer.weights}".replace("\n", "\n\t\t"))
    print("\n\tPerformance:")
    print(f"\t\t0-1 Loss = {loss}")
    print(f"\t\tF1 Score = {f1}")
    print("\n\n")

    # ----- DIFFERENTIAL EVOLUTION -----
    hp = {'num_replaced_parents': [1, 1, 4, 1], 'mutation_scale_factor': [1.6, 0.5, 2.5, 0.5],
          'crossover_rate': [0.2, 0.1, 0.3, 0.05], 'crossover': BinomialCrossover}
    hp_order = ['num_replaced_parents', 'mutation_scale_factor', 'crossover_rate']

    population_size = 30
    generations = 2
    tu = TuningUtility(DifferentialEvolution, training_test_folds, tuning_data, individual_eval_method, network_params, population_size, generations,
                       hp, hp_order)

    network, fitness = tu.train_on_fold(best_hp, training_data)


    fold_results = cv.calculate_results_for_fold(network, training_test_folds[0][1])

    loss = EvaluationMeasure.calculate_0_1_loss(fold_results)
    f1 = EvaluationMeasure.calculate_f_beta_score(fold_results, 2)

    print(f"Differential Evolution algorithm results on network with shape [35, 4], trained for 50 generations and a population of size 30")
    print(f"\tNetwork Weights:")
    for layer_num, layer in enumerate(network.layers):
        print(f"\t\tLayer {layer_num + 1}: \n{layer.weights}".replace("\n", "\n\t\t"))
    print("\n\tPerformance:")
    print(f"\t\t0-1 Loss = {loss}")
    print(f"\t\tF1 Score = {f1}")
    print("\n\n")


# def demonstrate_test_fold_output_regression():
#     pass

def demonstrate_GA_operations():
    print("------ Demonstrate Operations for the Genetic Algorithm ------")

    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    # Evaluation method
    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    method = EvaluationCallable(folds[0], individual_eval_method)

    print(
        "NOTE: We use 0_1 loss for our fitness function. In the following example, we computed the fitness for each chromosome against this fold from the Soybean dataset:")
    print(f"\n{folds[0]}")

    print("\nConsider the following population of four chromosomes:\n")
    # Generate five networks
    networks = []

    for i in range(4):
        nn = NeuralNetwork([35, 4], output_transformer, False, (-0.1, 0.1))
        serialized_nn = Utilities.serialize_network(nn)
        print(f"Network {i + 1}:", serialized_nn)
        networks.append(nn)

    print(
        "\nPerforming one round tournament selection with a tournament size of 4 yields the following chromosomes (parents):")

    select = TournamentSelect(method)
    parents = []

    for i in range(2):
        parent = select.select(networks, 2)
        parent_index = [idx for idx, el in enumerate(networks) if np.array_equal(el, parent)]
        del networks[parent_index[0]]
        parents.append(Utilities.serialize_network(parent))
        print(f"\nParent #{i + 1}:", Utilities.serialize_network(parent))

    print(
        "\n--- Performing uniform crossover with a cross rate of 0.8 on the selected parents yields the following children chromosomes:")
    print("**Notice about 80% of genes are crossed between the two chromosomes")

    cross = UniformCrossover()
    children = cross.cross(parents[0], parents[1])

    for i, child in enumerate(children):
        print(f"\nChild #{i + 1}:", child)

    print(
        "\n--- Performing uniform mutation with a mutation rate of 0.5 on the two children yields the following mutated children chromosomes: ---")
    print("**Notice about 50% of genes are mutated to a random value between -0.1 and 0.1")

    mutate = UniformMutation((-0.1, 0.1), 0.5)
    children = mutate.mutate(children)

    for i, child in enumerate(children):
        print(f"\nChild #{i + 1}:", child)


def demonstrate_DE_operations():
    print("------ Demonstrate Operations for the Differential Evolution Algorithm ------")

    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    # Evaluation method
    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    method = EvaluationCallable(folds[0], individual_eval_method)

    print(
        "NOTE: We use 0_1 loss for our fitness function. In the following example, we computed the fitness for each chromosome against this fold from the Soybean dataset:")
    print(f"\n{folds[0]}")

    print("\nConsider the following population of four chromosomes:\n")
    # Generate five networks
    networks = []

    for i in range(4):
        nn = NeuralNetwork([35, 4], output_transformer, False, (-0.1, 0.1))
        serialized_nn = Utilities.serialize_network(nn)
        print(f"Network {i + 1}:", serialized_nn)
        networks.append(serialized_nn)

    print("\nPerforming one round of DE selection with a selection size of 3 yields the following chromosomes:")
    target = networks[0]
    print(f"Target Chromosome: {target}\n")

    # Remove target chromosome from population
    del networks[0]

    # Select three chromosomes without replacement from population
    a, b, c = random.sample(networks, 3)

    print(f"Selected Chromosomes for Trial Vector:\nA: {a}\nB: {b}\nC: {c}")

    print(
        "\nPerforming differential mutation using trial = a + SCALE_FACTOR(b - c) with a scale factor of 0.5 yields a trial vector of:")
    trial_chromosome = a + (0.5 * (np.subtract(b, c)))
    print("Trial Vector:", trial_chromosome)

    print(
        "\n--- Performing binomial crossover with a cross rate of 0.8 on the target and trial vector yields the following trial vector:")
    print("**Notice about 80% of genes are crossed between the two chromosomes")

    cross = BinomialCrossover()
    trial_chromosome = cross.cross(target, trial_chromosome)

    print("Trial Vector:", trial_chromosome)


def demonstrate_PSO_operations():
    print("------ Demonstrate Operations for the Particle Swarm Optimization Algorithm ------")

    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    # Evaluation method
    def individual_eval_method(fold, network):
        return 1 - EvaluationMeasure.calculate_0_1_loss(cv.calculate_results_for_fold(network, fold))

    method = EvaluationCallable(folds[0], individual_eval_method)

    print(
        "NOTE: We use 0_1 loss for our fitness function. In the following example, we computed the fitness for each chromosome against this fold from the Soybean dataset:")
    print(f"\n{folds[0]}")

    print("\nConsider the following population of four chromosomes:\n")
    # Generate five networks
    networks = []

    for i in range(4):
        nn = NeuralNetwork([35, 4], output_transformer, False, (-0.1, 0.1))
        serialized_nn = Utilities.serialize_network(nn)
        print(f"Network {i + 1}:", serialized_nn)
        networks.append(nn)

    initial_best_position = Utilities.serialize_network(NeuralNetwork([35, 4], output_transformer, False, (-0.1, 0.1)))

    print("\nConsider the following particle:")
    particle = Particle(networks[0], method, 0.5, 1.496, 1.496)
    particle.pbest_position = initial_best_position
    particle.pbest_fitness = method(Utilities.deserialize_network(particle.original_network, particle.pbest_position))

    print(f"Particle Position: {Utilities.serialize_network(particle.original_network)}")
    print(f"Particle Inertia: {particle.inertia}")
    print(f"Particle Scale Factor 1 (c1): {particle.c1}")
    print(f"Particle Scale Factor 2 (c2): {particle.c2}")

    print("\nComputing the pbest calculation for the particle:")
    print("pbest is computed by comparing the fitness of the current particle position to the fitness of its best position")
    print(f"\nCurrent best position: {particle.pbest_position}")
    print(f"Fitness of current best position: {particle.pbest_fitness}")

    print(f"\nCurrent position: {particle.position}")
    current_position_fitness = method(Utilities.deserialize_network(particle.original_network, particle.position))
    print(f"Fitness of current position: {current_position_fitness}\n")

    if(current_position_fitness < particle.pbest_fitness):
        print("The fitness of the current position is better than the fitness of the best position, so we set the best position to the current position")
        particle.pbest_position = particle.position
    else:
        print("The fitness of the current position is worse than the fitness of the best position, so the best position remains unchanged")

    print(f"Updated pbest position: {particle.position}")

    print("\nComputing the gbest calculation for the particle:")
    initial_gbest_network = NeuralNetwork([35, 4], output_transformer, False, (-0.1, 0.1))
    initial_gbest_position = Utilities.serialize_network(initial_gbest_network)
    initial_gbest_fitness = method(initial_gbest_network)

    print("gbest is computed by comparing the fitness of the updated particles in the population with the best position for the entire population")

    print(f"\nCurrent global best position: {initial_gbest_position}")
    print(f"Fitness of current global best position: {initial_gbest_fitness}")

    print("Performing the pbest calculation for each particle yields:")

    best_particle = (0, math.inf)

    for index, network in enumerate(networks):
        print(f"\nParticle #{index + 1} pbest position: {Utilities.serialize_network(network)}")
        fitness = method(network)
        print(f"Particle #{index + 1} pbest fitness: {fitness}")

        if(fitness < best_particle[1]):
            best_particle = (index, fitness)


    print(f"The particle with the position with the best fitness was particle #{best_particle[0]} with a fitness of {best_particle[1]}")

    if (best_particle[1] < initial_gbest_fitness):
        print(
            f"The fitness of particle #{best_particle[0]} is better than the fitness of the current global best position, so we update the global best position")
        initial_gbest_position = Utilities.serialize_network(networks[best_particle[0]])
    else:
        print(
            f"The fitness of particle #{best_particle[0]} is worse than the fitness of the current global best position, so the global best position remains unchanged")

    print(f"Updated gbest position: {initial_gbest_position}")


    print("\nPerforming a velocity update for the particle:")
    particle.velocity = random.random() * particle.position
    print(f"Current velocity: {particle.velocity}")

    r1 = random.random()
    r2 = random.random()

    particle.pbest_position = initial_best_position
    particle.pbest_fitness = method(Utilities.deserialize_network(particle.original_network, particle.pbest_position))

    cognitive = particle.c1 + r1 * (initial_gbest_position - particle.position)
    social = particle.c2 + r2 * (particle.pbest_position - particle.position)

    print("\nComputing the cognitive component yields")
    print(f"Let r1 = {r1}")
    print(f"c1 + r1 * (global_best_position - current_position) = {particle.c1} + {r1} * (initial_gbest_position - particle.position)")
    print(f"Which equals: {cognitive}")

    print(f"\nComputing the social component yields")
    print(f"Let r2 = {r2}")
    print(f"c2 + r2 * (particle_best_position - current_position) = {particle.c2} + {r2} * (particle_best_position - particle.position)")
    print(f"Which equals: {social}")

    particle.velocity = particle.inertia * particle.velocity + cognitive + social

    print(f"\nThe updated velocity is then computed as inertia * velocity + cognitive + social")
    print(f"Which equals: {particle.velocity}")

    print("\nPerforming a position update for the particle:")
    print(f"Current position: {particle.position}")

    print(f"\nAdding the updated velocity to the current position yields the updated position")
    print(f"Updated position = particle_position + particle.velocity")
    print(f"Which equals: {particle.position + particle.velocity}")


def demonstrate_average_performance():
    with open(soybean_save_location, 'rb') as f:
        training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)

    print("---- Average Performance across Ten Folds for Each Algorithm ----")
    print("The following performance metrics were computed using these folds for the Soybean dataset:")

    for fold in folds:
        print(fold)

    algos = {
        "Genetic": {
            "[35, 4]": {"loss": "", "f1": ""},
            "[35, 5]": {"loss": "", "f1": ""},
            "[35, 6]": {"loss": "", "f1": ""}
        },
        "Differential Evolution": {
            "[35, 4]": {"loss": "", "f1": ""},
            "[35, 5]": {"loss": "", "f1": ""},
            "[35, 6]": {"loss": "", "f1": ""}
        },
        "Particle Swarm Optimization": {
            "[35, 4]": {"loss": "", "f1": ""},
            "[35, 5]": {"loss": "", "f1": ""},
            "[35, 6]": {"loss": "", "f1": ""}
        },
    }

    for algo, results in algos.items():
        for shape, metrics in results.items():
            print(f"\nAverage performance for the {algo} Algorithm on a network of shape {shape}")
            print(f"Loss: {metrics['loss']}")
            print(f"F1: {metrics['f1']}")


    # knn_cv = CrossValidation(training_data, target_feature="class")
    # knn_results = knn_cv.validate(KNN, 10, True, predict_params=[num_neighbors])

    # knn_measures = ExperimentHelper.convert_classification_results_to_measures(knn_results, classification_levels)
    # print("\nAverage Performance for Each Class (K-NN):")

    # for classification, performance in knn_measures.items():
    #     print(f"Class Level: {classification}\n{performance.mean()}\n")

    # EKNN
    # print("\nRunning on E-NN")
    # eknn_cv = CrossValidation(training_data, target_feature="class")
    # eknn_results = eknn_cv.validate(EditedKNN, 10, True, predict_params=[num_neighbors, False])

    # eknn_measures = ExperimentHelper.convert_classification_results_to_measures(eknn_results, classification_levels)
    # print("\nAverage Performance for Each Class (E-NN):")

    # for classification, performance in eknn_measures.items():
    #     print(f"Class Level: {classification}\n{performance.mean()}\n")

    # KMeans
    # print("\nRunning on K-Means")
    # kmeans_cv = CrossValidation(training_data, target_feature="class")
    # kmeans_results = kmeans_cv.validate(KNN, 10, True, model_params=[False, None, Minkowski(), True],
    #                                     predict_params=[num_neighbors])
    #
    # kmeans_measures = ExperimentHelper.convert_classification_results_to_measures(kmeans_results, classification_levels)
    # print("\nAverage Performance for Each Class (K-Means):")

    # for classification, performance in kmeans_measures.items():
    #     print(f"Class Level: {classification}\n{performance.mean()}\n")

def project_demonstration():
    demonstrate_test_fold_output_classification()
    input("")

    # demonstrate_test_fold_output_regression()
    # input("")

    # demonstrate_GA_operations()
    # input("")

    # demonstrate_DE_operations()
    # input("")

    # demonstrate_PSO_operations()
    # input("")

    # demonstrate_average_performance()
    # input("")


if __name__ == "__main__":
    project_demonstration()
