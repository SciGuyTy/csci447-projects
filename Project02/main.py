from BaseAlgorithm import BaseAlgorithm
import pandas as pd

def main():
    pass

def test_knn_on_breast_cancer():
    filepath = "../datasets/classification/BreastCancer/breast-cancer-wisconsin.data"
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

    data = pd.read_csv(filepath, names=column_headers, converters={"class": lambda x: (int(x) == 4)})
    rows_to_drop = [1057013, 1057013, 1096800, 1183246, 1184840, 1193683, 1197510, 1241232, 169356, 432809,
                    563649, 606140, 61634, 704168, 733639, 1238464, 1057067]

    data = data[~data['id'].isin(rows_to_drop)]
    knn = BaseAlgorithm(column_headers[1:-1], "class", data)

    print(knn.predict(data.iloc[0], 5))

if __name__ == "__main__":
    test_knn_on_breast_cancer()