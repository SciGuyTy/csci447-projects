from Preprocess import Preprocessor
from Algorithm import Algorithm
from Evaluation.Cross_validation import CrossValidation
from ExperimentHelper import ExperimentHelper

def run_breast_cancer_experiment():
    file_path = "../datasets/BreastCancer/breast-cancer-wisconsin.data"
    column_headers = ["id", "clump", "size", "shape", "adhesion", "epithelial_size", "nuclei", "chromatin", "nucleoli",
                      "mitoses", "class"]
    converters = {"class": lambda x: (int(x) == 4)}  # Convert the class column from ints to booleans
    bins = {"clump": 5, "size": 2}
    cols_to_drop = ['id']

    pp = Preprocessor()
    pp.load_raw_data_from_file(file_path, column_headers, columns_to_drop=cols_to_drop, converters=converters,
                               bins=bins)
    pp.save_processed_data_to_file("./breast_cancer_processed_data.csv")
    cv_unaltered = CrossValidation(pp.data, 'class', True)
    unaltered_results = cv_unaltered.validate(Algorithm, stratify=True)

    unaltered_measured_results = ExperimentHelper.convert_results_to_measures(unaltered_results)

    #print(unaltered_measured_results)
    #print(unaltered_measured_results.std())

    pp.alter_dataset(0.1)
    cv_altered = CrossValidation(pp.data, 'class', True)
    altered_results = cv_altered.validate(Algorithm, stratify=True)

    altered_measured_results = ExperimentHelper.convert_results_to_measures(altered_results)

    print(ExperimentHelper.run_t_tests_on_columns(unaltered_measured_results, altered_measured_results))

def run_congressional_voting_experiment():
    file_path = "../datasets/CongressionalVoting/house-votes-84.data"
    column_headers = ["party", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "epithelial_size", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
                      "aid-to-nicaraguan-contras", "mx-missile", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]

    pp = Preprocessor()
    pp.load_raw_data_from_file(file_path, column_headers)
    pp.save_processed_data_to_file("./congressional-votes-processed-data.csv")
    cv_unaltered = CrossValidation(pp.data, 'party', positive_class_value='democrat')
    unaltered_results = cv_unaltered.validate(Algorithm, stratify=True)

    unaltered_measured_results = ExperimentHelper.convert_results_to_measures(unaltered_results)

    #print(unaltered_measured_results)
    #print(unaltered_measured_results.std())

    pp.alter_dataset(0.1)
    cv_altered = CrossValidation(pp.data, 'party', positive_class_value='democrat')
    altered_results = cv_altered.validate(Algorithm, stratify=True)

    altered_measured_results = ExperimentHelper.convert_results_to_measures(altered_results)

    print(ExperimentHelper.run_t_tests_on_columns(unaltered_measured_results, altered_measured_results))


if __name__ == "__main__":
    run_congressional_voting_experiment()
