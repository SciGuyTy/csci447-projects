import datetime
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math

from source.Utilities.Utilities import CyclicalData, OneHotEncoding
from source.Utilities.ExperimentHelper import ExperimentHelper
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Algorithms.EditedKNN import EditedKNN
from source.Evaluation.CrossValidation import CrossValidation
from source.Algorithms.DistanceFunctions.ValueDifference import ValueDifference
from source.Utilities.Preprocess import Preprocessor
from source.Algorithms.KNN import KNN
from source.Utilities.TuningUtility import TuningUtility
from source.Evaluation.EvaluationMeasure import EvaluationMeasure
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import time

computer_hardware_save_folds_path = "../datasets/regression/ComputerHardware/folds.txt"
abalone_save_folds_path = "../datasets/regression/Abalone/folds.txt"
breast_cancer_save_folds_path = "../datasets/classification/BreastCancer/folds.txt"
soy_beans_save_folds_path = "../datasets/classification/Soybean/folds.txt"
forest_fires_save_folds_path = "../datasets/regression/ForestFires/folds.txt"
glass_identification_save_folds_path = "../datasets/classification/GlassIdentification/folds.txt"

'''
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
'''
'''
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

'''
def run_soybean_experiment(run_knn=False, run_eknn=False, run_kmeans=False):
    start_time = time.time()

    target_feature = "class"
    with open(soy_beans_save_folds_path, 'rb') as f:
        training_test_data, pp, training_data, tuning_data, cv = pickle.load(f)
    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)

    if run_knn:
        knn_start_time = time.time()
        tuning_utility = TuningUtility(KNN, pp.data, target_feature=target_feature)
        results = tuning_utility.tune_k_for_folds(training_test_data, tuning_data)

        tuned_params_knn =  {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in results.items()}

        print("Best params:", tuned_params_knn)
        print("KNN Tuning time:", time.time()-knn_start_time)

        final_raw_results_knn = cv.validate_for_folds(training_test_data, tuned_params_knn)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, 'D1') for i in final_raw_results_knn]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_knn]
        print("Final raw performance for knn", final_raw_results_knn)

        print("Final performance for knn (01, beta)", final_results_01, final_results_precision)

    if run_eknn:
        eknn_time = time.time()
        tuning_utility = TuningUtility(EditedKNN, pp.data, target_feature="class")
        results = tuning_utility.tune_k_for_folds(training_test_data, tuning_data, k_range=[1,5], train=True);
        tuned_params_eknn = {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in results.items()}

        print("Best params:", tuned_params_eknn)
        print("KNN Tuning time:", time.time() - eknn_time)

        final_raw_results_eknn = cv.validate_for_folds(training_test_data, tuned_params_eknn)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, 'D1') for i in final_raw_results_eknn]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_eknn]
        print("Final raw performance for eknn", final_raw_results_eknn)

        print("Final performance for eknn (01, beta)", final_results_01, final_results_precision)

        print("Best EKNN params", tuned_params_eknn)
        with open("soybeans-eknn-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_eknn, f)

        print("EKNN Tuning Time", time.time()-eknn_time)

    if run_kmeans:
        kmeans_time = time.time()
        with open("soybeans-eknn-best-params.bin", 'rb') as f:
            tuned_parameters_kmeans = pickle.load(f)

        clusters = [len(item['model'].training_data) for item in tuned_parameters_kmeans.values()]
        tuning_utility_kmeans = TuningUtility(KNN, pp.data, target_feature="class")
        all_params_kmeans = tuning_utility_kmeans.tune_k_for_folds(training_test_data, tuning_data, k_range=[1, 10],
                                                                   clusters=clusters)
        tuned_params_kmeans = {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in
                               all_params_kmeans.items()}

        print("Best params:", tuned_params_kmeans)
        print("KNN Tuning time:", time.time() - kmeans_time)

        final_raw_results_kmeans = cv.validate_for_folds(training_test_data, tuned_params_kmeans)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, 'D1') for i in final_raw_results_kmeans]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_kmeans]
        print("Final raw performance for kmeans", final_raw_results_kmeans)

        print("Final performance for kmeans (01, beta)", final_results_01, final_results_precision)

        print("Best kmeans params", tuned_params_kmeans)
        with open("kmeans-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_kmeans, f)

        print("kmeans Tuning Time", time.time() - kmeans_time)
        print("Final raw performance for kmeans", final_raw_results_kmeans)
        print("Final performance for eknn (01, beta)", final_results_01, final_results_precision)
    print("Total time:", time.time() - start_time)

def run_glass_expirement(run_knn=False, run_eknn=False, run_kmeans=False):
    start_time = time.time()

    target_feature = "class"
    with open(glass_identification_save_folds_path, 'rb') as f:
        training_test_data, pp, training_data, tuning_data, cv = pickle.load(f)
    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)

    if run_knn:
        knn_start_time = time.time()
        tuning_utility = TuningUtility(KNN, pp.data, target_feature=target_feature)
        results = tuning_utility.tune_k_for_folds(training_test_data, tuning_data)

        tuned_params_knn =  {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in results.items()}

        print("Best params:", tuned_params_knn)
        print("KNN Tuning time:", time.time()-knn_start_time)

        final_raw_results_knn = cv.validate_for_folds(training_test_data, tuned_params_knn)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, 1) for i in final_raw_results_knn]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_knn]
        print("Final raw performance for knn", final_raw_results_knn)

        print("Final performance for knn (01, beta)", final_results_01, final_results_precision)

    if run_eknn:
        eknn_time = time.time()
        tuning_utility = TuningUtility(EditedKNN, pp.data, target_feature="class")
        results = tuning_utility.tune_k_for_folds(training_test_data, tuning_data, k_range=[1,5], train=True);
        tuned_params_eknn = {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in results.items()}

        print("Best params:", tuned_params_eknn)
        print("KNN Tuning time:", time.time() - eknn_time)

        final_raw_results_eknn = cv.validate_for_folds(training_test_data, tuned_params_eknn)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, 1) for i in final_raw_results_eknn]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_eknn]
        print("Final raw performance for eknn", final_raw_results_eknn)

        print("Final performance for eknn (01, beta)", final_results_01, final_results_precision)

        print("Best EKNN params", tuned_params_eknn)
        with open("glass-eknn-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_eknn, f)

        print("EKNN Tuning Time", time.time()-eknn_time)

    if run_kmeans:
        kmeans_time = time.time()
        with open("glass-eknn-best-params.bin", 'rb') as f:
            tuned_parameters_kmeans = pickle.load(f)

        clusters = [len(item['model'].training_data) for item in tuned_parameters_kmeans.values()]
        tuning_utility_kmeans = TuningUtility(KNN, pp.data, target_feature="class")
        all_params_kmeans = tuning_utility_kmeans.tune_k_for_folds(training_test_data, tuning_data, k_range=[1, 10],
                                                                   clusters=clusters)
        tuned_params_kmeans = {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in
                               all_params_kmeans.items()}

        print("Best params:", tuned_params_kmeans)
        print("KNN Tuning time:", time.time() - kmeans_time)

        final_raw_results_kmeans = cv.validate_for_folds(training_test_data, tuned_params_kmeans)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, 1) for i in final_raw_results_kmeans]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_kmeans]
        print("Final raw performance for kmeans", final_raw_results_kmeans)

        print("Final performance for kmeans (01, beta)", final_results_01, final_results_precision)

        print("Best kmeans params", tuned_params_kmeans)
        with open("glass-kmeans-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_kmeans, f)

        print("kmeans Tuning Time", time.time() - kmeans_time)
        print("Final raw performance for kmeans", final_raw_results_kmeans)
        print("Final performance for eknn (01, beta)", final_results_01, final_results_precision)
    print("Total time:", time.time() - start_time)

def run_abalone_experiment(run_knn=False, run_eknn=False, run_kmeans=False):
    start_time = time.time()


    with open(abalone_save_folds_path, 'rb') as f:
        training_test_data, pp, training_data, tuning_data, cv = pickle.load(f)
    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)
    if run_knn:

        knn_start_time = time.time()
        tuning_utility = TuningUtility(KNN, pp.data, target_feature="rings", regression=True)
        all_results_knn = tuning_utility.tune_sigma_and_k_for_folds(training_test_data, tuning_data, [1, 1], 1, k_range=[1,10])
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
        all_params_eknn = tuning_utility.tune_sigma_k_and_epsilon_for_folds(training_test_data, tuning_data, [0.005, 0.005], 0.5, [0.5, 0.5], 0.5, train=True, k_range=[15,20]);
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
        tuning_utility_kmeans = TuningUtility(KNN, pp.data, target_feature="rings", regression=True)
        all_params_kmeans = tuning_utility_kmeans.tune_sigma_and_k_for_folds(training_test_data, tuning_data, [0.005, 0.005], 0.002, clusters=clusters, k_range=[1, 10])
        tuned_params_kmeans = TuningUtility.get_best_parameters_and_results(all_params_kmeans)
        print("Best K means params:", tuned_params_kmeans)
        with open("kmeans-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_kmeans, f)

        final_raw_results_kmeans = cv.validate_for_folds(training_test_data, tuned_params_kmeans)
        final_results_kmeans = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_kmeans]

        print("Final raw performance for kmeans", final_raw_results_kmeans)
        print("Final performance for kmeans", final_results_kmeans)

    print("Total time:", time.time()-start_time)


def run_breast_cancer_experiment(run_knn=False, run_eknn=False, run_kmeans=False):
    start_time = time.time()

    target_feature = "class"
    with open(breast_cancer_save_folds_path, 'rb') as f:
        training_test_data, pp, training_data, tuning_data, cv = pickle.load(f)
    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)

    if run_knn:
        knn_start_time = time.time()
        tuning_utility = TuningUtility(KNN, pp.data, target_feature=target_feature)
        results = tuning_utility.tune_k_for_folds(training_test_data, tuning_data)

        tuned_params_knn =  {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in results.items()}

        print("Best params:", tuned_params_knn)
        print("KNN Tuning time:", time.time()-knn_start_time)

        final_raw_results_knn = cv.validate_for_folds(training_test_data, tuned_params_knn)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, True) for i in final_raw_results_knn]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_knn]
        print("Final raw performance for knn", final_raw_results_knn)

        print("Final performance for knn (01, beta)", final_results_01, final_results_precision)

    if run_eknn:
        eknn_time = time.time()
        tuning_utility = TuningUtility(EditedKNN, pp.data, target_feature="class")
        results = tuning_utility.tune_k_for_folds(training_test_data, tuning_data, k_range=[1,5], train=True);
        tuned_params_eknn = {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in results.items()}

        print("Best params:", tuned_params_eknn)
        print("KNN Tuning time:", time.time() - eknn_time)

        final_raw_results_eknn = cv.validate_for_folds(training_test_data, tuned_params_eknn)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, True) for i in final_raw_results_eknn]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_eknn]
        print("Final raw performance for eknn", final_raw_results_eknn)

        print("Final performance for eknn (01, beta)", final_results_01, final_results_precision)

        print("Best EKNN params", tuned_params_eknn)
        with open("eknn-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_eknn, f)

        print("EKNN Tuning Time", time.time()-eknn_time)

    if run_kmeans:
        kmeans_time = time.time()
        with open("kmeans-best-params.bin", 'rb') as f:
            tuned_parameters_kmeans = pickle.load(f)

        clusters = [len(item['model'].training_data) for item in tuned_parameters_kmeans.values()]
        tuning_utility_kmeans = TuningUtility(KNN, pp.data, target_feature="class")
        all_params_kmeans = tuning_utility_kmeans.tune_k_for_folds(training_test_data, tuning_data, k_range=[1,6], clusters=clusters)
        tuned_params_kmeans = {fold: {'k': k, 'loss': loss, 'model': model} for fold, (k, loss, model) in all_params_kmeans.items()}

        print("Best params:", tuned_params_kmeans)
        print("KNN Tuning time:", time.time() - kmeans_time)

        final_raw_results_kmeans = cv.validate_for_folds(training_test_data, tuned_params_kmeans)
        final_results_precision = [EvaluationMeasure.calculate_f_beta_score(i, True) for i in final_raw_results_kmeans]
        final_results_01 = [EvaluationMeasure.calculate_0_1_loss(i) for i in final_raw_results_kmeans]
        print("Final raw performance for kmeans", final_raw_results_kmeans)

        print("Final performance for kmeans (01, beta)", final_results_01, final_results_precision)

        print("Best kmeans params", tuned_params_kmeans)
        with open("kmeans-best-params.bin", 'wb+') as f:
            pickle.dump(tuned_params_kmeans, f)

        print("kmeans Tuning Time", time.time() - kmeans_time)
        print("Final raw performance for kmeans", final_raw_results_kmeans)
        print("Final performance for eknn (01, beta)", final_results_01, final_results_precision)
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
        tuning_utility_kmeans = TuningUtility(KNN, pp.data, target_feature="ERP", regression=True)
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


def run_forest_fires_experiment(run_knn=False, run_eknn=False, run_kmeans=False):
    start_time = time.time()

    with open(forest_fires_save_folds_path, "rb") as f:
        training_test_data, pp, training_data, tuning_data, cv = pickle.load(f)
    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)

    if run_knn:
        knn_start_time = time.time()
        tuning_utility = TuningUtility(
            KNN, pp.data, target_feature="area", regression=True
        )
        all_results_knn = tuning_utility.tune_sigma_and_k_for_folds(
            training_test_data, tuning_data, [1, 10], 1
        )
        tuned_parameters_knn = TuningUtility.get_best_parameters_and_results(
            all_results_knn
        )

        print("Best results:", tuned_parameters_knn)
        print("KNN Tuning time:", time.time() - knn_start_time)

        final_raw_results_knn = cv.validate_for_folds(
            training_test_data, tuned_parameters_knn
        )
        final_results = [
            EvaluationMeasure.calculate_means_square_error(i)
            for i in final_raw_results_knn
        ]
        print("Final raw performance for knn", final_raw_results_knn)

        print("Final performance for knn", final_results)

    if run_eknn:
        eknn_time = time.time()
        tuning_utility = TuningUtility(
            EditedKNN, pp.data, target_feature="area", regression=True
        )
        all_params_eknn = tuning_utility.tune_sigma_k_and_epsilon_for_folds(
            training_test_data,
            tuning_data,
            [1, 3],
            1,
            [10, 20],
            5,
            train=True,
            k_range=[1, 12],
        )
        tuned_params_eknn = TuningUtility.get_best_parameters_and_results(
            all_params_eknn
        )
        print("Best EKNN params", tuned_params_eknn)
        with open("eknn-best-params.bin", "wb+") as f:
            pickle.dump(tuned_params_eknn, f)

        print("EKNN Tuning Time", time.time() - eknn_time)

        final_raw_results_eknn = cv.validate_for_folds(
            training_test_data, tuned_params_eknn
        )
        final_results = [
            EvaluationMeasure.calculate_means_square_error(i)
            for i in final_raw_results_eknn
        ]
        print("Final raw performance for eknn", final_raw_results_eknn)
        print("Final performance for eknn", final_results)

    if run_kmeans:
        with open("eknn-best-params.bin", "rb") as f:
            tuned_parameters_eknn = pickle.load(f)

        clusters = [
            len(item["model"].training_data) for item in tuned_parameters_eknn.values()
        ]
        tuning_utility_kmeans = TuningUtility(
            KNN, pp.data, target_feature="area", regression=True
        )

        all_params_kmeans = tuning_utility_kmeans.tune_sigma_and_k_for_folds(
            training_test_data, tuning_data, [0.005, 0.015], 0.001, clusters=clusters
        )
        tuned_params_kmeans = TuningUtility.get_best_parameters_and_results(
            all_params_kmeans
        )

        print("Best K means params:", tuned_params_kmeans)
        with open("kmeans-best-params.bin", "wb+") as f:
            pickle.dump(tuned_params_kmeans, f)

        final_raw_results_kmeans = cv.validate_for_folds(
            training_test_data, tuned_params_kmeans
        )
        final_results_kmeans = [
            EvaluationMeasure.calculate_means_square_error(i)
            for i in final_raw_results_kmeans
        ]
        print("Final raw performance for kmeans", final_raw_results_kmeans)
        print("Final performance for kmeans", final_results_kmeans)
    print("Total time:", time.time() - start_time)


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
    training_data = cv.get_tuning_set(0.25)
    cv = CrossValidation(training_data, "rings", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = training_data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "rings", regression=True)


    folded_training_data = cv.fold_data(10, True)
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

def create_folds_for_breast_cancer():

    file_path = "../datasets/classification/BreastCancer/breast-cancer-wisconsin.data"
    save_folds_path = breast_cancer_save_folds_path
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
    columns_to_floats = column_names[1:-1]

    pp = Preprocessor()

    # Define a converter that converts the class values into True or False
    feature_modifiers = {"class": lambda x: (int(x) == 4)}

    # Load the memory into data
    pp.load_raw_data_from_file(
        file_path,
        column_names,
        columns_to_drop=["id"],
        converters=feature_modifiers,
        dropNA=["?"],
        columns_to_floats =columns_to_floats
    )

    cv = CrossValidation(pp.data, "class")
    #training_data = cv.get_tuning_set(0.1)
    #cv = CrossValidation(training_data, "class")
    tuning_data = cv.get_tuning_set(0.1)
    #training_data = training_data.drop(tuning_data.index)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "class")

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    with open(save_folds_path, 'wb+') as f:
        pickle.dump([training_test_data, pp, training_data, tuning_data, cv], f)


def create_folds_for_soy_beans():

    file_path = "../datasets/classification/Soybean/soybean-small.data"
    save_folds_path = soy_beans_save_folds_path
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

    pp = Preprocessor()


    # Load the memory into data
    pp.load_raw_data_from_file(
        file_path,
        column_names,
        columns_to_drop=["date"],
    )

    cv = CrossValidation(pp.data, "class")
    #training_data = cv.get_tuning_set(0.1)
    #cv = CrossValidation(training_data, "class")
    tuning_data = cv.get_tuning_set(0.1)
    #training_data = training_data.drop(tuning_data.index)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "class")

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    with open(save_folds_path, 'wb+') as f:
        pickle.dump([training_test_data, pp, training_data, tuning_data, cv], f)

def create_folds_for_glass_identification():

    file_path = "../datasets/classification/GlassIdentification/glass.data"
    save_folds_path = glass_identification_save_folds_path
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

    cv = CrossValidation(pp.data, "class")
    #training_data = cv.get_tuning_set(0.1)
    #cv = CrossValidation(training_data, "class")
    tuning_data = cv.get_tuning_set(0.1)
    #training_data = training_data.drop(tuning_data.index)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "class")

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    with open(save_folds_path, 'wb+') as f:
        pickle.dump([training_test_data, pp, training_data, tuning_data, cv], f)



def create_folds_for_forest_fires():
    pp = Preprocessor()

    file_path = "../datasets/regression/ForestFires/forestfires.data"
    save_folds_path = forest_fires_save_folds_path
    column_names = [
        "x",
        "y",
        "month",
        "day",
        "ffmc",
        "dmc",
        "dc",
        "isi",
        "temp",
        "rh",
        "wind",
        "rain",
        "area",
        ]
    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    day_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}

    feature_modifiers = {"month": lambda x: CyclicalData(month_map[x], 12), "day": lambda x: CyclicalData(day_map[x], 7)}

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        converters=feature_modifiers
    )
    cv = CrossValidation(pp.data, "area", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "area", regression=True)

    folded_training_data = cv.fold_data(2, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    with open(save_folds_path, "wb+") as f:
        pickle.dump([training_test_data, pp, training_data, tuning_data, cv], f)

if __name__ == "__main__":
    print("Starting at:", datetime.datetime.now())
    # create_folds_for_abalone()
    # run_abalone_experiment(False, True, True)
    #create_folds_for_computer_hard()
    #run_computer_hardware_experiment(run_kmeans=True)

    run_glass_expirement(True, True, True)
    #run_breast_cancer_experiment(False, False, True)
    # create_folds_for_forest_fires()
    #run_forest_fires_experiment(False, True, False)


