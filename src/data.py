import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def download_data() -> None:
    data_dir = "../data/diabetes"

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        os.system(
            "curl -L -o ../data/diabetes.zip https://www.kaggle.com/api/v1/datasets/download/brandao/diabetes"
        )
        os.system("unzip ../data/diabetes.zip -d ../data/diabetes")
        os.system("rm ../data/diabetes.zip")
    else:
        print("Dataset already exists, skipping download.")
