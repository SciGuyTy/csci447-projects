import math
from source.Utilities.ExperimentHelper import ExperimentHelper
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Algorithms.EditedKNN import EditedKNN
from source.Evaluation.CrossValidation import CrossValidation
from source.Algorithms.DistanceFunctions.ValueDifference import ValueDifference
from source.Utilities.Preprocess import Preprocessor
from source.Algorithms.KNN import KNN
import pandas as pd


def main():
    pass


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

    # cv = CrossValidation(pp.data.sample(100), "ERP", regression=True)
    # print(cv.validate(KNN, 10, True, predict_params=[2]))

    eknn = EditedKNN(training_data=pp.data.sample(100), target_feature="ERP", regression=True, h=10, sigma=10)
    print(eknn.predict(pp.data.iloc[0], 5))

if __name__ == "__main__":
    run_breast_cancer_experiment()
    # run_soybean_experiment()
    # test_knn_on_ch()
