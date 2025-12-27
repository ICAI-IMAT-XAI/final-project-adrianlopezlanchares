from typing import List

import torch
from sklearn.ensemble import RandomForestClassifier


class MLPModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(MLPModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(torch.nn.BatchNorm1d(h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=0.5))
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
