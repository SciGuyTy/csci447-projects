from Project04.NeuralNetwork import NeuralNetwork


class TuningUtility:

    def __init__(self, algorithm, folds, tuning_data, evaluation_method, network_parameters, population_size,
                 generations, hyperparameters, hyperparamters_tuning_order):
        self.algorithm = algorithm

        self.folds = folds

        self.tuning_data = tuning_data

        self.evaluation_method = evaluation_method

        self.network_parameters = network_parameters

        self.population_size = population_size
        self.generations = generations

        # Dictionary of hyperparameters.
        # Key = string name of hyperparameter
        # Value = [untuned_value, start_value, final_value, step_size]
        self.hyperparameters = hyperparameters
        self.tuning_order = hyperparamters_tuning_order

    def tune_hyperparameters(self):
        best_hp = self.hyperparameters

        # Loop through the hyperparameters and add the initial value to the best_hp
        for hp in self.tuning_order:
            best_hp[hp] = self.hyperparameters[hp][0]

        # Tune each value one at a time in the given order
        for hp in self.tuning_order:
            untuned_value, start_value, final_value, step_size = self.hyperparameters[hp]

            # Keep track of the performances for each value
            performances = dict()

            # Loop through each of the possible values
            for value in range(start_value, final_value, step_size):
                best_hp[hp] = value

                network_performances_sum = 0.0

                for fold in self.folds:
                    # Create a population of networks
                    networks = []
                    for _ in range(self.population_size):
                        networks.append(NeuralNetwork(self.network_parameters['shape'],
                                                      self.network_parameters['output_transformer'],
                                                      self.network_parameters['regression'],
                                                      self.network_parameters['random_weight_range']))


                    alg = self.algorithm(networks, best_hp, self.evaluation_method)
                    # Train the algorithm and get the best network and fitness
                    network, fitness = alg.train(self.generations)

                    tuning_data_fitness = self.tuning_evaluation
                    # Sum the fitnesses
                    network_performances_sum += fitness


                # Add the average performance of the hyperparameters to the dict
                performances[value] = network_performances_sum / len(self.folds)

            # Keep the best hyperparameter
            best_hp[hp] = min(performances, key=performances.get)

        return best_hp


