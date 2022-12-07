import math
import pickle
import random

import numpy as np
import pandas as pd

from Project04.Algorithms.Crossover.BinomialCrossover import BinomialCrossover
from Project04.Algorithms.Crossover.UniformCrossover import UniformCrossover
from Project04.Algorithms.Mutation.UniformMutation import UniformMutation
from Project04.Algorithms.PSO import Particle
from Project04.Algorithms.Selection.TournamentSelect import TournamentSelect
from Project04.Evaluation.EvaluationCallable import EvaluationCallable
from Project04.Evaluation.EvaluationMeasure import EvaluationMeasure
from Project04.NeuralNetwork import NeuralNetwork
from Project04.Utilities.Utilities import Utilities

soybean_save_location = "./ExperimentSaves/soybean.objects"


def output_transformer(output_vector: np.array):
    return output_vector.argmax() + 1


# def demonstrate_test_fold_output_classification():
#     print("------ Demonstrate Output/Performance of Test Folds ------")
#
#     soybean_save_location = "./ExperimentSaves/soybean.objects"
#
#     with open(soybean_save_location, 'rb') as f:
#         training_test_folds, PP, tuning_data, folds, training_test_folds, cv = pickle.load(f)
#
#     classes = PP.data["class"].unique()
#
#     def classification_modifier(pattern: pd.Series):
#         target = [0] * len(classes)
#         target[pattern.loc["class"] - 1] = 1
#         return target
#
#     print("Consider a classification network for the Soybean dataset:")
#     for num_layers in range(3):
#         layers = num_layers
#         training_params = {
#             "learning_rate": 0.01,
#             "momentum": 0.1,
#             "batch_size": 1,
#             "epochs": 1,
#             "initial_weight_range": (-.1, 0.1)
#         }
#
#         tu = TuningUtility(training_test_folds, tuning_data, "class", 35, 4, classification_modifier, output_transformer, True, training_params)
#         best_models = tu.tune_for_h_hidden_layers(layers)
#
#         output = cv.validate_for_folds(training_test_folds, best_models)
#
#         loss = [EvaluationMeasure.calculate_0_1_loss(i) for i in output]
#         f1 = [EvaluationMeasure.calculate_f_beta_score(i, 1) for i in output]
#         print(f"Test Fold #1 from Soybean CV with {num_layers} Hidden Layers")
#         print(f"\tFold: \n{training_test_folds[0][1]}".replace("\n", "\n\t\t"))
#         print("")
#         print(f"\tOutput: \n{output[0]}".replace("\n", "\n\t\t"))
#         print("\n\tPerformance:")
#         print(f"\t\t0-1 Loss = {loss[0]}")
#         print(f"\t\tF1 Score = {f1[0]}")
#         print("\n\n")
#
# def demonstrate_test_fold_output_regression():
#     abalone_save_location = './ExperimentSaves/abalone.objects'
#
#     with open(abalone_save_location, 'rb') as f:
#         training_test_data, pp, tuning_data, folded_training_data, cv = pickle.load(f)
#
#     def regression_modifier(pattern: pd.Series):
#         return [pattern['rings']]
#
#     print("\nConsider a regression network for the Abalone dataset:")
#     for num_layers in range(3):
#         layers = num_layers
#         training_params = {
#             "learning_rate": 0.001,
#             "momentum": 0.01,
#             "batch_size": 5,
#             "epochs": 1,
#             "initial_weight_range": (-0.1, 0.1)
#         }
#
#         tu = TuningUtility(training_test_data, tuning_data, "rings", 8, 1, regression_modifier,
#                            regression_output_transformer, False, training_params)
#         best_models = tu.tune_for_h_hidden_layers(layers)
#
#         output = cv.validate_for_folds(training_test_data, best_models)
#
#         mse = [EvaluationMeasure.calculate_means_square_error(i) for i in output]
#
#         print(f"Test Fold #1 from Breast Cancer CV with {num_layers} Hidden Layers")
#         print(f"\tFold: \n{training_test_data[0][1]}".replace("\n", "\n\t\t"))
#         print("")
#         print(f"\tOutput: \n{output[0]}".replace("\n", "\n\t\t"))
#         print("\n\tPerformance:")
#         print(f"\t\tMSE = {mse[0]}")
#         print("\n\n")

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


# def demonstrate_average_performance():
#     pass

def project_demonstration():
    # demonstrate_test_fold_output_classification()
    # input("")
    #
    # demonstrate_test_fold_output_regression()
    # input("")

    # demonstrate_GA_operations()
    # input("")

    # demonstrate_DE_operations()
    # input("")

    demonstrate_PSO_operations()
    input("")

    # demonstrate_average_performance()
    # input("")


if __name__ == "__main__":
    project_demonstration()
