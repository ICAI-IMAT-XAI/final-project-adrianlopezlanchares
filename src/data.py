import os

import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
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


def build_preprocessor() -> ColumnTransformer:
    """Builds a SKLearn preprocessor for the Diabetes dataset

    Returns:
        ColumnTransformer: Unfitted preprocessor
    """

    # Define column types
    ordinal_cols = ["age", "weight", "A1Cresult", "max_glu_serum"]
    binary_cols = ["change", "diabetesMed"]
    categorical_cols = [
        "race",
        "gender",
        "payer_code",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    # Define orders
    age_order = [
        "[0-10)",
        "[10-20)",
        "[20-30)",
        "[30-40)",
        "[40-50)",
        "[50-60)",
        "[60-70)",
        "[70-80)",
        "[80-90)",
        "[90-100)",
    ]
    weight_order = [
        "?",
        "[0-25)",
        "[25-50)",
        "[50-75)",
        "[75-100)",
        "[100-125)",
        "[125-150)",
        "[150-175)",
        "[175-200)",
        ">200",
    ]
    a1c_order = ["None", "Norm", ">7", ">8"]
    glu_order = ["None", "Norm", ">200", ">300"]

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[age_order, weight_order, a1c_order, glu_order]
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "ord",
                ordinal_transformer,
                ordinal_cols,
            ),
            ("bin", OneHotEncoder(drop="if_binary"), binary_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="passthrough",
    )

    return preprocessor


class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
