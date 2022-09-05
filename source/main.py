from Preprocess import Preprocessor
from Algorithm import Algorithm
from Evaluation.Cross_validation import CrossValidation
from Evaluation.EvaluationMeasure import EvaluationMeasure as EM
from ExperimentHelper import ExperimentHelper


import pandas as pd
from scipy.stats import ttest_ind


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



if __name__ == "__main__":
    run_breast_cancer_experiment()
