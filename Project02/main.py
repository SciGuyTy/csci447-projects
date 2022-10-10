import datetime
import pickle

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


def test_knn_on_breast_cancer():
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

    # knn = KNN(training_data=pp.data.sample(100), target_feature="class")
    # print(knn.predict(pp.data.iloc[0], 5))

    # cv = CrossValidation(pp.data.sample(100), "class")
    # print(cv.validate(KNN, 10, True, predict_params=[2]))

    eknn = EditedKNN(training_data=pp.data.sample(100), target_feature="class")
    print(eknn.edit_and_predict(pp.data.iloc[0], 5, reduce_redundancy=False))


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

def run_computer_hardware_expirement():
    start_time = time.time()


    with open(computer_hardware_save_folds_path, 'rb') as f:
        training_test_data, pp, training_data, tuning_data, cv = pickle.load(f)

    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)
    '''tuning_utility = TuningUtility(KNN, pp.data, target_feature="ERP", regression=True)


    all_results_knn = tuning_utility.tune_sigma_and_k_for_folds(training_test_data, tuning_data, [1, 20], 1)
    tuned_parameters_knn = TuningUtility.get_best_parameters_and_results(all_results_knn)

    print("Best results:", tuned_parameters_knn)
    print("KNN Tuning time:", time.time()-pp_time)

    final_raw_results_knn = cv.validate_for_folds(KNN, training_test_data, tuned_parameters_knn)
    final_results = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_knn]
    print("Final raw performance for knn", final_raw_results_knn)

    print("Final performance for knn", final_results)
'''
    eknn_time = time.time()

    tuning_utility = TuningUtility(EditedKNN, pp.data, target_feature="ERP", regression=True)
    all_params_eknn = tuning_utility.tune_sigma_k_and_epsilon_for_folds(training_test_data, tuning_data, [0.005, 0.015], 0.002, [1,3], 1);
    tuned_params_eknn = TuningUtility.get_best_parameters_and_results(all_params_eknn)
    print("Best Results", tuned_params_eknn)
    with open("eknn-best-params.bin", 'wb+') as f:
        pickle.dump(tuned_params_eknn, f)

    print("EKNN Tuning Time", time.time()-eknn_time)

    final_raw_results_eknn = cv.validate_for_folds(training_test_data, tuned_params_eknn)
    final_results = [EvaluationMeasure.calculate_means_square_error(i) for i in final_raw_results_eknn]
    print("Final raw performance for knn", final_raw_results_eknn)
    print("Final performance for knn", final_results)

    print("Total time:", time.time()-start_time)



def create_folds_for_computer_hard():

    file_path = "../datasets/regression/ComputerHardware/machine.data"
    save_folds_path = "../datasets/regression/ComputerHardware/machine.data"
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

    cv = CrossValidation(pp.data, "ERP", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "ERP", regression=True)

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    with open(computer_hardware_save_folds_path, 'wb+') as f:
        pickle.dump([training_test_data, pp, training_data, tuning_data, cv], f)


if __name__ == "__main__":
    print("Starting at:", datetime.datetime.now())
    run_computer_hardware_expirement()


