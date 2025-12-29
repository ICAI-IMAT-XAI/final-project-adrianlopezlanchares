import os

import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from torch.utils.data import Dataset


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


class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
