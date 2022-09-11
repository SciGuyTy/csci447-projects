from Preprocess import Preprocessor
from Algorithm import Algorithm
from Evaluation.CrossValidation import CrossValidation
from ExperimentHelper import ExperimentHelper


def run_breast_cancer_experiment():
    file_path = "../datasets/BreastCancer/breast-cancer-wisconsin.data"
    column_headers = [
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
    converters = {
        "class": lambda x: (int(x) == 4)
    }  # Convert the class column from ints to booleans
    cols_to_drop = ["id"]

    pp = Preprocessor()
    pp.load_raw_data_from_file(
        file_path, column_headers, columns_to_drop=cols_to_drop, converters=converters
    )
    pp.save_processed_data_to_file("./breast_cancer_processed_data.csv")
    cv = CrossValidation(pp.data, "class")
    unaltered_results = cv.validate(Algorithm, stratify=True)

    unaltered_measured_results = ExperimentHelper.convert_results_to_measures(
        unaltered_results, [True]
    )

    cv_altered = CrossValidation(pp.data, "class")
    altered_results = cv_altered.validate(Algorithm, stratify=True, alter_data=True)

    altered_measured_results = ExperimentHelper.convert_results_to_measures(
        altered_results, [True]
    )

    print(
        ExperimentHelper.run_t_tests_on_columns(
            unaltered_measured_results[True], altered_measured_results[True]
        )
    )


def run_congressional_voting_experiment():
    file_path = "../datasets/CongressionalVoting/house-votes-84.data"
    column_headers = [
        "party",
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "epithelial_size",
        "el-salvador-aid",
        "religious-groups-in-schools",
        "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missile",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa",
    ]

    pp = Preprocessor()
    pp.load_raw_data_from_file(file_path, column_headers)
    pp.save_processed_data_to_file("./congressional-votes-processed-data.csv")
    cv = CrossValidation(pp.data, "party", positive_class_value="democrat")
    unaltered_results = cv.validate(Algorithm, stratify=True)

    unaltered_measured_results = ExperimentHelper.convert_results_to_measures(
        unaltered_results, ["democrat"]
    )

    altered_results = cv.validate(Algorithm, stratify=True, alter_data=True)
    altered_measured_results = ExperimentHelper.convert_results_to_measures(
        altered_results, ["democrat"]
    )

    print(
        ExperimentHelper.run_t_tests_on_columns(
            unaltered_measured_results["democrat"], altered_measured_results["democrat"]
        )
    )


def run_iris_experiment():
    # Define properties about the dataset
    file_path = "../datasets/Iris/iris.data"
    column_names = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
        "class",
    ]

    # Bin the continuous values
    attribute_bins = {
        "sepal_length_cm": 5,
        "sepal_width_cm": 5,
        "petal_length_cm": 5,
        "petal_width_cm": 5,
    }

    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(file_path, column_names, bins=attribute_bins)
    PP.save_processed_data_to_file(
        "../datasets/Iris/iris-processed.csv"
    )

    classes = PP.data["class"].unique()

    # Dictionary to store experimental results for each classification level
    experimental_results = dict.fromkeys(PP.data["class"].unique())

    # Compute performance metrics
    # for classification in PP.data["class"].unique():
    PP.load_processed_data_from_file(
        "../datasets/Iris/iris-processed.csv"
    )
    
    # Compute performance metrics for unaltered data
    CV_unaltered = CrossValidation(PP.data)
    unaltered_results = CV_unaltered.validate(Algorithm, 10, stratify=True)
    unaltered_metrics = ExperimentHelper.convert_results_to_measures(unaltered_results, classes)

    # Compute performance metrics for altered data
    CV_altered = CrossValidation(PP.data)
    altered_results = CV_altered.validate(Algorithm, 10, stratify=True, alter_data=True)
    altered_metrics = ExperimentHelper.convert_results_to_measures(altered_results, classes)

    for classification in PP.data["class"].unique():
        # Compare the unaltered and altered performance results
        experimental_results[classification] = ExperimentHelper.run_t_tests_on_columns(
            unaltered_metrics[classification], altered_metrics[classification]
        )

    # Display experiment results
    print(ExperimentHelper.format_results(experimental_results))


def run_soybean_experiment():
    file_path = "../datasets/Soybean/soybean-small.data"
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

    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(file_path, column_names)
    PP.save_processed_data_to_file("../datasets/Soybean/soybean-small-processed.csv")

    classes = PP.data["class"].unique()

    # Dictionary to store experimental results for each classification level
    experimental_results = dict.fromkeys(PP.data["class"].unique())

    # Compute performance metrics for unaltered data
    CV_unaltered = CrossValidation(PP.data)
    unaltered_results = CV_unaltered.validate(Algorithm, 10, stratify=True)
    unaltered_metrics = ExperimentHelper.convert_results_to_measures(unaltered_results, classes)

    # Compute performance metrics for altered data
    CV_altered = CrossValidation(PP.data)
    altered_results = CV_altered.validate(Algorithm, 10, stratify=True, alter_data=True)
    altered_metrics = ExperimentHelper.convert_results_to_measures(altered_results, classes)

    for classification in PP.data["class"].unique():
        # Compare the unaltered and altered performance results
        experimental_results[classification] = ExperimentHelper.run_t_tests_on_columns(
            unaltered_metrics[classification], altered_metrics[classification]
        )

    # Display experiment results
    print(ExperimentHelper.format_results(experimental_results))


def run_glass_identification_experiment():
    file_path = "../datasets/GlassIdentification/glass.data"
    column_names = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]

    attribute_bins = {
        "RI": 5,
        "Na": 5,
        "Mg": 5,
        "Al": 5,
        "Si": 5,
        "K": 5,
        "Ca": 5,
        "Ba": 5,
        "Fe": 5,
    }

    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(
        file_path, column_names, bins=attribute_bins, columns_to_drop=["ID"]
    )
    
    PP.save_processed_data_to_file(
        "../datasets/GlassIdentification/glass-processed.data"
    )

    classes = PP.data["class"].unique()

    # Dictionary to store experimental results for each classification level
    experimental_results = dict.fromkeys(PP.data["class"].unique())

    # Compute performance metrics for unaltered data
    CV_unaltered = CrossValidation(PP.data)
    unaltered_results = CV_unaltered.validate(Algorithm, 10, stratify=True)
    unaltered_metrics = ExperimentHelper.convert_results_to_measures(unaltered_results, classes)

    # Compute performance metrics for altered data
    CV_altered = CrossValidation(PP.data)
    altered_results = CV_altered.validate(Algorithm, 10, stratify=True, alter_data=True)
    altered_metrics = ExperimentHelper.convert_results_to_measures(altered_results, classes)

    for classification in PP.data["class"].unique():
        # Compare the unaltered and altered performance results
        experimental_results[classification] = ExperimentHelper.run_t_tests_on_columns(
            unaltered_metrics[classification], altered_metrics[classification]
        )

    # Display experiment results
    print(ExperimentHelper.format_results(experimental_results))


if __name__ == "__main__":
    print("\nRunning Iris Experiment")
    run_iris_experiment()
    # print("\nRunning Soybean Experiment")
    # run_soybean_experiment()
    # print("\nRunning Glass Identification Experiment")
    # run_glass_identification_experiment()
    # print("\nRunning BC Experiment")
    # run_breast_cancer_experiment()
