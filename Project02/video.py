from random import sample
from source.Algorithms.KMeans import KMeans
from source.Utilities.Utilities import Utilities
from source.Utilities.ExperimentHelper import ExperimentHelper
from source.Algorithms.EditedKNN import EditedKNN
from source.Algorithms.KNN import KNN
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Evaluation.CrossValidation import CrossValidation
from source.Utilities.Preprocess import Preprocessor

import pandas as pd

pp = Preprocessor()


def load_soybean_data():
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
    pp.load_raw_data_from_file(file_path, column_names)


def load_glass_identification_data():
    # Define the filepath to dataset
    file_path = "../datasets/classification/GlassIdentification/glass.data"

    # Define the features
    column_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

    # Load the memory into data
    pp.load_raw_data_from_file(file_path, column_names)


def load_machine_data():
    # Define the filepath to dataset
    file_path = "../datasets/regression/ComputerHardware/machine.data"

    # Define the features
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

    # Load the memory into data
    pp.load_raw_data_from_file(
        file_path, column_names, columns_to_drop=["vendor_name", "model_name"]
    )



def demonstrate_folding():
    load_soybean_data()
    cv = CrossValidation(pp.data, "class")

    print("---- Soybean Dataset being split into ten folds ----")
    print("Original soybean dataset:\n", pp.data)

    print("\nFolded soybean dataset (using stratification):\n")
    folds = cv.fold_data(10, True)

    for index, fold in enumerate(folds):
        print(f"Fold #{index + 1}")
        print(fold)


def demonstrate_distance_calculation():
    load_machine_data()

    minkowski = Minkowski()
    samples = pp.data.sample(2)
    features = pp.data.columns.drop("ERP")

    norm_params = Utilities.normalize(pp.data, "ERP")

    print("---- Calculating Distances ----")

    print("Consider sample #1 and sample #2 from the Glass Identification dataset...")
    print("Sample 1\n", samples.iloc[0].to_frame().T)
    print("\nSample 2\n", samples.iloc[1].to_frame().T)

    Utilities.normalize_set_by_params(samples, norm_params)

    distance = minkowski.compute_distance(
        samples.iloc[0][features], samples.iloc[1][features]
    )

    print(
        "Normalizing and passing them into our Euclidean configuration of the Minkowski metric yields a distance of:\n",
        distance,
    )

    return distance


def demonstrate_kernel_calculation(distance, sigma):
    knn = KNN(pp.data, "ERP", False)
    kernel_value = knn.gaussian_kernel(distance, 1 / sigma)

    print("---- Calculating Kernel ----")
    print(
        f"Using the distance of {distance} provided from these previous samples and a sigma of {sigma}, the Gaussian Kernel function returns:\n",
        kernel_value,
    )


def demonstrate_knn_classification(k):
    load_soybean_data()

    instance = pp.data.sample(1)
    training_data = pp.data.drop(instance.index)

    knn = KNN(training_data, "class", False)

    print("---- Performing Classification Using KNN ----")
    print(
        "Consider a holdout point for the Soybean dataset (removed from training data):"
    )
    print(instance)

    print(
        f"\nPerforming k-NN for k = {k} neighbors yields the following nearest neighbors and distances:"
    )

    neighbors = knn.find_k_nearest_neighbors(instance.iloc[0], k)

    for index, (d, sample) in enumerate(neighbors):
        print(
            f"\nNeighbor #{index + 1}\n Distance: {d}\n Sample: {sample.to_frame().T}\n"
        )

    prediction = knn.predict(instance.iloc[0], k)

    print("The classification results of performing KNN on the instance are:")
    print(f"Actual Class: {instance.iloc[0]['class']}\nPredicted Class: {prediction}")


def demonstrate_knn_regression(k, sigma):
    load_machine_data()

    # Get a sample from the data for classification, and remove it from the training data
    instance = pp.data.sample(1)
    training_data = pp.data.drop(instance.index)

    # Make a copy of the instance before normalization
    original_instance = instance.copy()

    # Normalize the training data and instance data
    norm_params = Utilities.normalize(pp.data, "ERP")
    Utilities.normalize_set_by_params(training_data, norm_params)
    Utilities.normalize_individual_by_params(instance, norm_params)

    knn = KNN(training_data, "ERP", True, sigma, None)

    print("---- Performing Regression Using KNN ----")
    print(
        "Consider a holdout point for the Computer Hardware dataset (removed from training data):"
    )
    print(original_instance)

    print(
        f"\nNormalizing the data and performing k-NN for k = {k} neighbors and sigma = {sigma} yields the following nearest neighbors and distances:"
    )

    # Record the nearest neighbors
    neighbors = knn.find_k_nearest_neighbors(instance.iloc[0], k)

    # Print out the k nearest neighbors
    for index, (d, sample) in enumerate(neighbors):
        print(
            f"\nNeighbor #{index + 1}\n Distance: {d}\n Sample: {sample.to_frame().T}\n"
        )

    # Make a regression prediction on the instance
    prediction = knn.predict(instance.iloc[0], k)

    print("The regression results of performing KNN on the instance are:")
    print(f"Actual ERP: {instance.iloc[0]['ERP']}\nPredicted ERP: {prediction}")


def demonstrate_editing():
    load_soybean_data()

    enn = EditedKNN(pp.data.copy(), "class")

    print("---- Editing a Sample from Training Data Using ENN ----")

    print(
        "Consider the number of samples in the original Soybean training data being passed into Edited KNN:"
    )
    print(len(enn.training_data))

    print(
        "\nRunning one editing iteration of KNN edits out noisy sample points such that the new length of the training data is:"
    )
    
    # Perform one iteration of editing
    enn._minimize_data(5, False, None)
    print(len(enn.training_data))

    print("\nThe following samples were edited out of the original training data:")
    print(pd.concat([pp.data, enn.training_data]).drop_duplicates(keep=False))


def demonstrate_clustering(k):
    load_soybean_data()

    # Copy data into a training set
    training_data = pp.data.copy()
    sample_point = training_data.sample(1)

    kmeans = KMeans("class", training_data, k)

    print("---- Associating a Point with a Cluster in K-Means ----")
    print("\nConsider the following point from the training set passed into the KMeans Clustering algorithm:")

    print(sample_point)

    print(f"\nRunning one iteration of K-Means clustering with k={k} clusters results in the following clusters:")

    # Perform one iteration of clustering
    kmeans._initialize_clusters()
    kmeans._update_samples(False)

    # Print clusters
    for index, cluster in enumerate(kmeans.clusters):
        print(f"\nCluster #{index}")
        print(f"Centroid")
        print(cluster["centroid"].to_frame().T)

        print(f"\nSamples")
        print(cluster["samples"])

def demonstrate_average_classification_performance():
    load_soybean_data()

    training_data = pp.data.copy()

    print(
        "---- Average Classification Performance across Ten Folds for Each Algorithm ----"
    )

    # Retrieve classification levels
    classification_levels = training_data["class"].unique()

    # # Perform 10-fold CV for K-NN
    # print("\nRunning on K-NN")
    # # Define hyper-parameters
    # num_neighbors = 5

    # knn_cv = CrossValidation(training_data, target_feature="class")
    # knn_results = knn_cv.validate(KNN, 10, True, predict_params=[num_neighbors])

    # # Convert the results to performance metrics
    # knn_measures = ExperimentHelper.convert_classification_results_to_measures(
    #     knn_results, classification_levels
    # )
    # print("\nAverage Performance for Each Class (K-NN):")

    # # Report average performance from CV
    # for classification, performance in knn_measures.items():
    #     print(f"Class Level: {classification}\n{performance.mean()}\n")

    # # Perform 10-fold CV for E-NN
    # print("\nRunning on E-NN")
    # # Define hyper-parameters
    # num_neighbors = 5

    # eknn_cv = CrossValidation(training_data, target_feature="class")
    # eknn_results = eknn_cv.validate(EditedKNN, 10, True, predict_params=[num_neighbors, False, 0.0])

    # # Convert the results to performance metrics
    # eknn_measures = ExperimentHelper.convert_classification_results_to_measures(
    #     eknn_results, classification_levels
    # )
    # print("\nAverage Performance for Each Class (E-NN):")

    # # Report average performance from CV
    # for classification, performance in eknn_measures.items():
    #     print(f"Class Level: {classification}\n{performance.mean()}\n")

    # Perform 10-fold CV for K-Means
    print("\nRunning on K-Means")
    # Define hyper-parameters
    num_neighbors = 5
    num_clusters = 5

    kmeans_cv = CrossValidation(training_data, target_feature="class")
    kmeans_results = kmeans_cv.validate(
        KNN, 10, True, model_params=[False, None, num_clusters], predict_params=[num_neighbors]
    )

    # Convert the results to performance metrics
    kmeans_measures = ExperimentHelper.convert_classification_results_to_measures(
        kmeans_results, classification_levels
    )
    print("\nAverage Performance for Each Class (K-Means):")

    # Report average performance from CV
    for classification, performance in kmeans_measures.items():
        print(f"Class Level: {classification}\n{performance.mean()}\n")


def demonstrate_average_regression_performance():
    load_machine_data()
    training_data = pp.data.copy()

    print(
        "---- Average Regression Performance across Ten Folds for Each Algorithm ----"
    )

    # Perform 10-fold CV for K-NN
    print("\nRunning on K-NN")
    # Define hyper-parameters
    num_neighbors = 5
    sigma = 5

    knn_cv = CrossValidation(training_data, target_feature="ERP", regression=True)
    knn_results = knn_cv.validate(
        KNN, 10, True, model_params=[True, sigma, None], predict_params=[num_neighbors]
    )

    # Convert the results to performance metrics
    knn_measures = ExperimentHelper.convert_regression_results_to_measures(knn_results)
    
    # Report average performance from CV
    print("\nAverage Performance across folds (K-NN):")
    print(knn_measures.mean())

    # Perform 10-fold CV for E-NN
    print("\nRunning on E-NN")
    # Define hyper-parameters
    num_neighbors = 5
    sigma = 5
    err_threshold = 1.0

    eknn_cv = CrossValidation(training_data, target_feature="ERP", regression=True)
    eknn_results = eknn_cv.validate(
        EditedKNN,
        10,
        True,
        model_params=[True, sigma, None],
        predict_params=[num_neighbors, False, err_threshold],
    )

    # Convert the results to performance metrics
    eknn_measures = ExperimentHelper.convert_regression_results_to_measures(
        eknn_results
    )

    # Report average performance from CV
    print("\nAverage Performance across folds (E-NN):")
    print(eknn_measures.mean())

    # Perform 10-fold CV for K-Means
    print("\nRunning on K-Means")
    # Define hyper-parameters
    num_neighbors = 5
    sigma = 5

    kmeans_cv = CrossValidation(training_data, target_feature="ERP", regression=True)
    kmeans_results = kmeans_cv.validate(
        KNN, 10, True, model_params=[True, sigma, True], predict_params=[num_neighbors]
    )

    # Convert the results to performance metrics
    kmeans_measures = ExperimentHelper.convert_regression_results_to_measures(
        kmeans_results
    )

    # Report average performance from CV
    print("\nAverage Performance across folds (K-Means):")
    print(kmeans_measures.mean())


def project_demonstration():
    # # Show the Soybean dataset being split into ten folds
    # demonstrate_folding()
    # input("")

    # # Demonstrate the calculation of the Minkowski (Euclidean) Distance Function
    # distance = demonstrate_distance_calculation()
    # input("")

    # # Demonstrate the calculation of the Kernel Function
    # demonstrate_kernel_calculation(distance, 10)
    # input("")

    # # Demonstrate of k-NN classification for a data point (show point and neighbors)
    # demonstrate_knn_classification(5)
    # input("")

    # # Demonstrate k-NN regression for a data point (show point and neighbors)
    # demonstrate_knn_regression(5, 0.01)
    # input("")

    # # Demonstrate editing out a data point using Edited k-NN
    # demonstrate_editing()
    # input("")

    # # Demonstrate a data point be associated with a cluster in k-Means Clustering
    # demonstrate_clustering(3)
    # input("")

    # Display average performance across ten folds for k-NN, k-Means, and ENN on a classification data set
    demonstrate_average_classification_performance()
    input("")

    # Display average performance across ten folds for k-NN, k-Means, and ENN on a regression data set
    # demonstrate_average_regression_performance()


if __name__ == "__main__":
    project_demonstration()
