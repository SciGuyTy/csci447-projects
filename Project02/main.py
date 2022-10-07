from source.Evaluation.CrossValidation import CrossValidation
from source.Algorithms.DistanceFunctions.ValueDifference import ValueDifference
from source.Utilities.Preprocess import Preprocessor
from source.Algorithms.KNN import KNN
from source.Utilities.TuningUtility import TuningUtility
import pandas as pd
import time

def main():
    pass


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
        columns_to_normalize=column_names[1:-1],
        converters=feature_modifiers,
        dropNA=["?"],
    )

    # knn = KNN(training_data=pp.data.sample(100), target_feature="class")
    # print(knn.predict(pp.data.iloc[0], 5))

    cv = CrossValidation(pp.data.sample(100), "class")
    print(cv.validate(KNN, 10, True, predict_params=[2]))

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
        "ERP"
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

def test_tuning_utility_regression():
    start_time = time.time()

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
        "ERP"
    ]

    pp = Preprocessor()

    pp.load_raw_data_from_file(
        file_path,
        column_names,
        columns_to_drop=["vendor_name", "model_name"],
    )

    pp_time = time.time()
    print("Preprocessing time:", pp_time - start_time)
    cv = CrossValidation(pp.data, "ERP", regression=True)
    tuning_data = cv.get_tuning_set(0.1)
    training_data = pp.data.drop(tuning_data.index)
    cv = CrossValidation(training_data, "ERP", regression=True)

    folded_training_data = cv.fold_data(10, True)
    training_test_data = cv.get_training_test_data_from_folds(folded_training_data)

    tuning_utility = TuningUtility(KNN, pp.data, target_feature="ERP", regression=True)


    all_results = tuning_utility.tune_sigma_and_k_for_folds(training_test_data, tuning_data, [1, 20], 1)
    best_results = TuningUtility.get_best_parameters_and_results(all_results)

    print("Best results:", best_results)
    print("Tuning time:", time.time()-pp_time)
    print("Total time:", time.time()-start_time)


if __name__ == "__main__":
    # test_knn_on_breast_cancer()
    test_tuning_utility_regression()
