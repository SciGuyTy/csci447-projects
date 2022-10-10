from source.Utilities.ExperimentHelper import ExperimentHelper
from source.Algorithms.EditedKNN import EditedKNN
from source.Algorithms.KNN import KNN
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Evaluation.CrossValidation import CrossValidation
from source.Utilities.Preprocess import Preprocessor

import pandas as pd

pp = Preprocessor()

def load_soybean_data():
    #Define the filepath to dataset
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

def load_glass_data():
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

def load_machine_data():
    #Define the filepath to dataset
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
        file_path,
        column_names,
        columns_to_drop=["vendor_name", "model_name"]
    )

def demonstrate_folding():
    load_soybean_data()
    cv = CrossValidation(pp.data, "class")

    print("---- Soybean Dataset being split into ten folds ----")
    print("Original soybean dataset:\n", pp.data)

    print("\nFolded soybean dataset (using stratification):\n")
    folds = cv.fold_data(10, True)

    for index, fold in enumerate(folds):
        print(f"\nFold #{index + 1}")
        print(fold)

def demonstrate_distance_calculation():
    load_glass_data()
    minkowski = Minkowski()
    samples = pp.data.sample(2)

    glass_features = pp.data.columns.drop("class")
    
    distance = minkowski.compute_distance(samples.iloc[0][glass_features], samples.iloc[1][glass_features])

    print("---- Calculating Distances ----")

    print("Consider sample #1 and sample #2 from the Glass Identification dataset...")
    print("Sample 1")
    print(samples.iloc[0].to_frame().T)
    print("\nSample 2")
    print(samples.iloc[1].to_frame().T)   

    print("\nPassing them into our Euclidean configuration of the Minkowski metric yields a distance of:")
    print(distance)

    return distance

def demonstrate_kernel_calculation(distance, sigma):
    knn = KNN(pp.data, "class", False)
    kernel_value = knn.gaussian_kernel(distance, 1 / sigma)

    print("---- Calculating Kernel ----")
    print(f"Using the distance of {distance} provided from these previous samples and a sigma of {sigma}, the Gaussian Kernel function returns:")
    print(kernel_value)

def demonstrate_knn_classification(k):
    load_soybean_data()
    instance = pp.data.sample(1)
    training_data = pp.data.drop(instance.index)

    knn = KNN(training_data, "class", regression=False)

    print("---- Performing Classification Using KNN ----")
    print("Consider a holdout point for the soybean dataset (removed from training data):")
    print(instance)

    print(f"\nPerforming k-NN for k = {k} neighbors yields the following nearest neighbors and distances:")
    
    neighbors = knn.find_k_nearest_neighbors(instance.iloc[0], k)
    
    for index, (d, sample) in enumerate(neighbors):
        print(f"\nNeighbor #{index + 1}\n Distance: {d}\n Sample: {sample.to_frame().T}\n")

    prediction = knn.predict(instance.iloc[0], k)

    print("The classification results of performing KNN on the instance are:")
    print(f"Actual Class: {instance.iloc[0]['class']}\nPredicted Class: {prediction}")

def demonstrate_knn_regression(k, sigma):
    load_machine_data()
    instance = pp.data.sample(1)
    training_data = pp.data.drop(instance.index)

    knn = KNN(training_data, "ERP", regression=True, sigma=sigma)

    print("---- Performing Regression Using KNN ----")
    print("Consider a holdout point for the computer hardware dataset (removed from training data):")
    print(instance)

    print(f"\nPerforming k-NN for k = {k} neighbors and sigma = {sigma} yields the following nearest neighbors and distances:")
    
    neighbors = knn.find_k_nearest_neighbors(instance.iloc[0], k)
    
    for index, (d, sample) in enumerate(neighbors):
        print(f"\nNeighbor #{index + 1}\n Distance: {d}\n Sample: {sample.to_frame().T}\n")

    prediction = knn.predict(instance.iloc[0], k)

    print("The regression results of performing KNN on the instance are:")
    print(f"Actual ERP: {instance.iloc[0]['ERP']}\nPredicted ERP: {prediction}")

def demonstrate_editing():
    load_soybean_data()
    enn = EditedKNN(pp.data.copy(), "class")
    
    print("---- Editing a Sample from Training Data Using ENN ----")

    print("Consider the number of samples in the original soybean training data being passed into Edited KNN:")
    print(len(enn.training_data))

    print("\nRunning one editing iteration of KNN edits out noisy sample points such that the new length of the training data is:")
    enn._minimize_data(5, False, None)
    print(len(enn.training_data))

    print("\nThe following samples were edited out of the original training data:")
    print(pd.concat([pp.data, enn.training_data]).drop_duplicates(keep=False))

def demonstrate_clustering():
    pass

def demonstrate_average_classification_performance(num_neighbors, num_clusters):
    load_glass_data()
    training_data = pp.data.copy()

    print("---- Average Classification Performance across Ten Folds for Each Algorithm ----")
    print("The following algorithms are performed on the Glass Identification dataset ")

    classification_levels = training_data["class"].unique()

    # KNN
    # print(f"\nRunning on K-NN (k = {num_neighbors})")
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
    print("\nRunning on K-Means")
    kmeans_cv = CrossValidation(training_data, target_feature="class")
    kmeans_results = kmeans_cv.validate(KNN, 10, True, model_params=[False, None, Minkowski(), True], predict_params=[num_neighbors])

    kmeans_measures = ExperimentHelper.convert_classification_results_to_measures(kmeans_results, classification_levels)
    print("\nAverage Performance for Each Class (K-Means):")
    
    for classification, performance in kmeans_measures.items():
        print(f"Class Level: {classification}\n{performance.mean()}\n")

def demonstrate_average_regression_performance(sigma, num_neighbors, num_clusters, epsilon):
    load_machine_data()
    training_data = pp.data.copy().sample(10)

    print("---- Average Regression Performance across Ten Folds for Each Algorithm ----")
    print("The following algorithms are performed on the Computer Hardware dataset ")

    # KNN
    print(f"\nRunning on K-NN (sigma = {sigma}, k = {num_neighbors}")
    knn_cv = CrossValidation(training_data, target_feature="ERP", regression=True)
    knn_results = knn_cv.validate(KNN, 10, True, model_params=[True, sigma], predict_params=[num_neighbors])

    knn_measures = ExperimentHelper.convert_regression_results_to_measures(knn_results)
    print("\nAverage Performance across folds (K-NN):")
    print(knn_measures.mean())

    # EKNN
    print(f"\nRunning on E-NN (sigma = {sigma}, k = {num_neighbors}, epsilon = {epsilon})")
    eknn_cv = CrossValidation(training_data, target_feature="ERP", regression=True)
    eknn_results = eknn_cv.validate(EditedKNN, 10, True, model_params=[False, True, sigma, epsilon, num_neighbors], predict_params=[num_neighbors, False])

    eknn_measures = ExperimentHelper.convert_regression_results_to_measures(eknn_results)
    print("\nAverage Performance across folds (E-NN):")
    print(eknn_measures.mean())

    # KMeans
    print(f"\nRunning on K-Means (sigma = {sigma}, k = {num_neighbors}, num_clusters = {num_clusters})")
    kmeans_cv = CrossValidation(training_data, target_feature="ERP", regression=True)
    kmeans_results = kmeans_cv.validate(KNN, 10, True, model_params=[True, sigma, Minkowski(), True], predict_params=[num_neighbors])

    kmeans_measures = ExperimentHelper.convert_regression_results_to_measures(kmeans_results)
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

    # Demonstrate a data point be associated with a cluster in k-Means Clustering

    # Display average performance across ten folds for k-NN, k-Means, and ENN on a classification data set
    demonstrate_average_classification_performance(5, 5)
    input("")

    # Display average performance across ten folds for k-NN, k-Means, and ENN on a regression data set
    demonstrate_average_regression_performance(5, 5, 5, 1)

if __name__ == "__main__":
    project_demonstration()