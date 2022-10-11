import datetime
import pickle

import math

from Project02.source.Utilities.Utilities import OneHotEncoding
from source.Utilities.ExperimentHelper import ExperimentHelper
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Algorithms.EditedKNN import EditedKNN
from source.Evaluation.CrossValidation import CrossValidation
from source.Algorithms.DistanceFunctions.ValueDifference import ValueDifference
from source.Utilities.Preprocess import Preprocessor
from source.Algorithms.KNN import KNN
from source.Utilities.TuningUtility import TuningUtility
from source.Evaluation.EvaluationMeasure import EvaluationMeasure
import pandas as pd
import time

computer_hardware_save_folds_path = "../datasets/regression/ComputerHardware/folds.txt"
abalone_save_folds_path = "../datasets/regression/Abalone/folds.txt"


def project_demonstration():
    # Show the Soybean dataset being split into ten folds

    # Demonstrate the calculation of the Minkowski (Euclidean) Distance Function

    # Demonstrate the calculation of the Kernel Function

    # Demonstrate of k-NN classification for a data point (show point and neighbors)

    # Demonstrate k-NN regression for a data point (show point and neighbors)

    # Demonstrate editing out a data point using Edited k-NN

    # Demonstrate a data point be associated with a cluster in k-Means Clustering

    # Display average performance across ten folds for k-NN, k-Means, and ENN on a classification data set

    # Display average performance across ten folds for k-NN, k-Means, and ENN on a regression data set
    pass

def run_breast_cancer_experiment():
    ### Preprocess ###
    # Create an instance of the preprocessor utility to facilitate data wrangling
    pp = Preprocessor()

    # Define the filepath to dataset
    file_path = "../datasets/classification/BreastCancer/breast-cancer-wisconsin.data"

    # Define the features
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
    feature_modifiers = {"class": lambda x: (int(x) == 4)}

    # Load the memory into data
    pp.load_raw_data_from_file(
        file_path,
        column_names,
        columns_to_drop=["id"],
        converters=feature_modifiers,
        dropNA=["?"],
    )

    ### Tune Hyper Parameters ####

    # TODO: Implement tuning utility
    training_data = pp.data.sample(40)
    tuned_k = 5

    ### Evaluate Model Performance ####
    # Retrieve classification levels
    classification_levels = training_data["class"].unique()

    # KNN
    knn_cv = CrossValidation(training_data, target_feature="class")
    knn_results = knn_cv.validate(KNN, 10, True, predict_params=[tuned_k])
    knn_measures = ExperimentHelper.convert_results_to_measures(knn_results, classification_levels)

    # EKNN
    eknn_cv = CrossValidation(training_data, target_feature="class")
    eknn_results = eknn_cv.validate(EditedKNN, 10, True, predict_params=[tuned_k, False, 0.0])
    eknn_measures = ExperimentHelper.convert_results_to_measures(eknn_results, classification_levels)

    # KMeans
    kmeans_cv = CrossValidation(training_data, target_feature="class")
    kmeans_results = kmeans_cv.validate(KNN, 10, True, model_params=[True], predict_params=[tuned_k])
    kmeans_measures = ExperimentHelper.convert_results_to_measures(kmeans_results, classification_levels)
    
    print("---------- Results for Breast Cancer Experiment ----------")
    print(f"KNN:\n{knn_measures}")

def run_glass_identification_experiment():
    ### Preprocess ###
    # Create an instance of the preprocessor utility to facilitate data wrangling
    pp = Preprocessor()

    # Define the filepath to dataset
    file_path = "../datasets/classification/GlassIdentification/glass.data"

    # Define the features
    column_names = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]

    # Load the memory into data
    pp.load_raw_data_from_file(
        file_path,
        column_names,
        columns_to_drop=["ID"]
    )

    ### Tune Hyper Parameters ####

    # TODO: Implement tuning utility
    training_data = pp.data.sample(math.floor(0.9 * len(pp.data)))
    tuned_k = 5

    ### Evaluate Model Performance ####
    # KNN
    knn_cv = CrossValidation(training_data, target_feature="class")
    knn_results = knn_cv.validate(KNN, 10, True, predict_params=[tuned_k])

    print(knn_results)

    # EKNN
    eknn_cv = CrossValidation(training_data, target_feature="class")
    eknn_results = eknn_cv.validate(EditedKNN, 10, True, predict_params=[tuned_k, False, 0.0])

    print(eknn_results)

    # KMeans
    kmeans_cv = CrossValidation(training_data, target_feature="class")
    kmeans_results = kmeans_cv.validate(KNN, 10, True, model_params=[True], predict_params=[tuned_k, False, 0.0])

    print(kmeans_results)

def run_soybean_experiment():
    ### Preprocess ###
    # Create an instance of the preprocessor utility to facilitate data wrangling
    pp = Preprocessor()

    # Define the filepath to dataset
    file_path = "../datasets/classification/Soybean/soybean-small.data"

    # Define the features
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

    # Load the memory into data
    pp.load_raw_data_from_file(
        file_path,
        column_names
    )

    ### Tune Hyper Parameters ####

    # TODO: Implement tuning utility
    training_data = pp.data.sample(math.floor(0.9 * len(pp.data)))
    tuned_k = 5

    ### Evaluate Model Performance ####
    classification_levels = training_data["class"].unique()

    knn_cv = CrossValidation(training_data, target_feature="class")
    knn_results = knn_cv.validate(KNN, 10, True, predict_params=[tuned_k])

    knn_measures = ExperimentHelper.convert_results_to_measures(knn_results, classification_levels)
    print(knn_measures)

    # EKNN
    eknn_cv = CrossValidation(training_data, target_feature="class")
    eknn_results = eknn_cv.validate(EditedKNN, 10, True, predict_params=[tuned_k, False, 0.0])

    eknn_measures = ExperimentHelper.convert_results_to_measures(eknn_results)
    print(eknn_measures)

    # KMeans
    kmeans_cv = CrossValidation(training_data, target_feature="class")
    kmeans_results = kmeans_cv.validate(KNN, 10, True, model_params=[True], predict_params=[tuned_k])

    kmeans_measures = ExperimentHelper.convert_results_to_measures(kmeans_results)
    print(kmeans_measures)

def test_knn_on_ch():
    file_path = "../datasets/regression/ComputerHardware/machine.data"
    column_names = [
        "vendor_name",
        "model_name",
        "MYCT",
        "MMIN",
        "MMAX",
        "CACH",
        "CHMIN",
        "CHMAX",
        "PRP",
        "ERP",
    ]

    pp = Preprocessor()

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        columns_to_drop=["vendor_name", "model_name"],
    )

    # knn = KNN(training_data=pp.data.sample(100), target_feature="class")
    # print(knn.predict(pp.data.iloc[0], 5))

    cv = CrossValidation(pp.data.sample(100), "ERP", regression=True)
    print(cv.validate(KNN, 10, True, predict_params=[2]))

def test_tuning_utility():
    start_time = time.time()
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

    pp = Preprocessor()

    feature_modifiers = {"class": lambda x: (int(x) == 4)}

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        columns_to_drop=["id"],
        converters=feature_modifiers,
        dropNA=["?"],
    )
    pp_time = time.time()
    print("Preprocessing time: ", pp_time - start_time)
    cv = CrossValidation(pp.data, "class")
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "class")

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)
    tu = TuningUtility(KNN, pp.data)

    best_k = tu.tune_k_for_folds(training_test_data, tuning_data)
    print("Best K value:", best_k)
    print("Tuning time:", time.time()-pp_time)
    print("Total time:", time.time()-start_time)

def run_abalone_experiment(run_knn=False, run_eknn=False, run_kmeans=False):
    start_time = time.time()


    with open(abalone_save_folds_path, 'rb') as f:
        training_test_data, pp, training_data, tuning_data, cv = pickle.load(f)
    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)
    if run_knn:

        knn_start_time = time.time()
        tuning_utility = TuningUtility(KNN, pp.data, target_feature="rings", regression=True)
        all_results_knn = tuning_utility.tune_sigma_and_k_for_folds(training_test_data, tuning_data, [1, 3], 1, k_range=[1,10])
        tuned_parameters_knn = TuningUtility.get_best_parameters_and_results(all_results_knn)

        print("Best results:", tuned_parameters_knn)
        print("KNN Tuning time:", time.time()-knn_start_time)

        final_raw_results_knn = cv.validate_for_folds(training_test_data, tuned_parameters_knn)
        final_results = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_knn]
        print("Final raw performance for knn", final_raw_results_knn)

        print("Final performance for knn", final_results)


    if run_eknn:
        eknn_time = time.time()
        tuning_utility = TuningUtility(EditedKNN, pp.data, target_feature="rings", regression=True)
        all_params_eknn = tuning_utility.tune_sigma_k_and_epsilon_for_folds(training_test_data, tuning_data, [1, 3], 1, [1,3], 1, train=True);
        tuned_params_eknn = TuningUtility.get_best_parameters_and_results(all_params_eknn)
        print("Best EKNN params", tuned_params_eknn)
        with open("eknn-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_eknn, f)

        print("EKNN Tuning Time", time.time()-eknn_time)

        final_raw_results_eknn = cv.validate_for_folds(training_test_data, tuned_params_eknn)
        final_results = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_eknn]
        print("Final raw performance for eknn", final_raw_results_eknn)
        print("Final performance for eknn", final_results)

    if run_kmeans:
        with open("eknn-best-params.bin", 'rb') as f:
            tuned_parameters_eknn = pickle.load(f)

        clusters = [len(item['model'].training_data) for item in tuned_parameters_eknn.values()]
        tuning_utility_kmeans = TuningUtility(KNN, pp.data, target_feature="ERP", regression=True)
        all_params_kmeans = tuning_utility_kmeans.tune_sigma_and_k_for_folds(training_test_data, tuning_data, [0.005, 0.015], 0.002, clusters=clusters)
        tuned_params_kmeans = TuningUtility.get_best_parameters_and_results(all_params_kmeans)
        print("Best K means params:", tuned_params_kmeans)
        with open("kmeans-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_kmeans, f)

        final_raw_results_kmeans = cv.validate_for_folds(training_test_data, tuned_params_kmeans)
        final_results_kmeans = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_kmeans]

        print("Final raw performance for kmeans", final_raw_results_kmeans)
        print("Final performance for kmeans", final_results_kmeans)

    print("Total time:", time.time()-start_time)


def run_computer_hardware_experiment(run_knn=False, run_eknn=False, run_kmeans=False):
    start_time = time.time()


    with open(computer_hardware_save_folds_path, 'rb') as f:
        training_test_data, pp, training_data, tuning_data, cv = pickle.load(f)
    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)

    if run_knn:
        knn_start_time = time.time()
        tuning_utility = TuningUtility(KNN, pp.data, target_feature="ERP", regression=True)
        all_results_knn = tuning_utility.tune_sigma_and_k_for_folds(training_test_data, tuning_data, [0.001, 0.010], 0.002)
        tuned_parameters_knn = TuningUtility.get_best_parameters_and_results(all_results_knn)

        print("Best results:", tuned_parameters_knn)
        print("KNN Tuning time:", time.time()-knn_start_time)

        final_raw_results_knn = cv.validate_for_folds(training_test_data, tuned_parameters_knn)
        final_results = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_knn]
        print("Final raw performance for knn", final_raw_results_knn)

        print("Final performance for knn", final_results)

    if run_eknn:
        eknn_time = time.time()
        tuning_utility = TuningUtility(EditedKNN, pp.data, target_feature="ERP", regression=True)
        all_params_eknn = tuning_utility.tune_sigma_k_and_epsilon_for_folds(training_test_data, tuning_data, [0.005, 0.015], 0.002, [1,4], 1, train=True);
        tuned_params_eknn = TuningUtility.get_best_parameters_and_results(all_params_eknn)
        print("Best EKNN params", tuned_params_eknn)
        with open("eknn-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_eknn, f)

        print("EKNN Tuning Time", time.time()-eknn_time)

        final_raw_results_eknn = cv.validate_for_folds(training_test_data, tuned_params_eknn)
        final_results = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_eknn]
        print("Final raw performance for eknn", final_raw_results_eknn)
        print("Final performance for eknn", final_results)

    if run_kmeans:
        with open("eknn-best-params.bin", 'rb') as f:
            tuned_parameters_eknn = pickle.load(f)

        clusters = [len(item['model'].training_data) for item in tuned_parameters_eknn.values()]
        tuning_utility_kmeans = TuningUtility(KNN, pp.data, target_feature="rings ", regression=True)
        all_params_kmeans = tuning_utility_kmeans.tune_sigma_and_k_for_folds(training_test_data, tuning_data, [0.005, 0.015], 0.001, clusters=clusters)
        tuned_params_kmeans = TuningUtility.get_best_parameters_and_results(all_params_kmeans)
        print("Best K means params:", tuned_params_kmeans)
        with open("kmeans-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_kmeans, f)

        final_raw_results_kmeans = cv.validate_for_folds(training_test_data, tuned_params_kmeans)
        final_results_kmeans = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_kmeans]
        print("Final raw performance for kmeans", final_raw_results_kmeans)
        print("Final performance for kmeans", final_results_kmeans)
    print("Total time:", time.time()-start_time)


def create_folds_for_abalone():
    file_path = '../datasets/regression/Abalone/abalone.data'
    save_folds_path = abalone_save_folds_path

    column_names = [
        'sex',
        'length',
        'diameter',
        'height',
        'whole_weight',
        'shucked_weight',
        'viscera_weight',
        'shell_weight',
        'rings',
    ]
    feature_modifiers = {"sex": lambda x: OneHotEncoding(x)}
    pp = Preprocessor()

    pp.load_raw_data_from_file(file_path, column_names, converters=feature_modifiers)

    cv = CrossValidation(pp.data, "rings", regression=True)
    tuning_data = cv.get_tuning_set(0.005)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "rings", regression=True)
    sample_data = cv.get_tuning_set(0.05)

    training_data = sample_data
    cv = CrossValidation(sample_data, "rings", regression=True)

    folded_training_data = cv.fold_data(2, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    with open(save_folds_path, 'wb+') as f:
        pickle.dump([training_test_data, pp, training_data, tuning_data, cv], f)


def create_folds_for_computer_hard():

    file_path = "../datasets/regression/ComputerHardware/machine.data"
    save_folds_path = computer_hardware_save_folds_path
    column_names = [
        "vendor_name",
        "model_name",
        "MYCT",
        "MMIN",
        "MMAX",
        "CACH",
        "CHMIN",
        "CHMAX",
        "PRP",
        "ERP"
    ]

    pp = Preprocessor()

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        columns_to_drop=["vendor_name", "model_name"],
    )
    # Convert all the columns to float (needed to allow one hot encoding for other data sets)
    pp.data = pp.data.astype('float64')

    cv = CrossValidation(pp.data, "ERP", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "ERP", regression=True)

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    with open(save_folds_path, 'wb+') as f:
        pickle.dump([training_test_data, pp, training_data, tuning_data, cv], f)


if __name__ == "__main__":
    print("Starting at:", datetime.datetime.now())
    create_folds_for_abalone()
    run_abalone_experiment(False, True, True)
    #create_folds_for_computer_hard()
    #run_computer_hardware_experiment(run_kmeans=True)

