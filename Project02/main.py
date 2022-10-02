from sqlite3 import converters
from source.Utilities.Preprocess import Preprocessor
from source.Algorithms.KNN import KNN
import pandas as pd


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

    knn = KNN(column_names[1:-1], "class", pp.data)
    print(knn.predict(pp.data.iloc[0], 5))


if __name__ == "__main__":
    test_knn_on_breast_cancer()
