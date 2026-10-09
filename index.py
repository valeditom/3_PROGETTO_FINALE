import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.datasets import load_diabetes

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV,
    learning_curve
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. CARICAMENTO E ABALISI DEL DATASET
# ============================================================

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

df = pd.DataFrame(
    X,
    columns=diabetes.feature_names
)

df['target'] = y

# print(df)

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

# ============================================================
# 2. VISUALIZZAZIONE TRAMITE STRUMENTI GRAFICI
# ============================================================

## BOXPLOT
df.boxplot(figsize=(14, 6))
plt.xticks(rotation=45)
plt.title("Boxplot - Diabetes Dataset")
plt.show()

## ISTOGRAMMI
df.hist(
figsize=(14, 8))
plt.suptitle("Distribuzione delle Feature")
plt.tight_layout()
plt.show()

## SCATTER MATRIX
scatter_matrix(df, figsize=(14, 14))
plt.suptitle("Scatter Matrix - Diabetes Dataset")
plt.show()


# ============================================================
# 2. PREPARAZIONE DATI IN TRAINING E TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
