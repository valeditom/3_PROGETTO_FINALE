import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_diabetes
from sklearn.model_selection import (train_test_split, KFold, cross_val_score, GridSearchCV, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (mean_squared_error, r2_score)

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
# 3. PREPARAZIONE DATI IN TRAINING E TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# 4. DEFINIZIONE MODELLI CON PIPELINE
# =========================================

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeRegressor(random_state=42))
    ]),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model",Ridge())
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso())
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor())
    ]),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR())
    ])
}

# =========================================
# 5. K-FOLD CROSS VALIDATION
# =========================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_mean_squared_error"
    )
    nmse_mean = scores.mean()
    mse_mean = -nmse_mean
    results.append([
        name,
        nmse_mean,
        mse_mean
    ])

# =========================================
# 6. TABELLA RISULTATI
# =========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "NMSE",
        "MSE"
    ]
)
results_df = results_df.sort_values(by="MSE")
print(results_df)

# =========================================
# 7. GRID SEARCH SU RIDGE
# =========================================

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

param_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    cv=kf,
    scoring="neg_mean_squared_error"
)

grid_search.fit(
    X_train,
    y_train
)

print("Migliori parametri:", grid_search.best_params_)

print("Miglior NMSE:", grid_search.best_score_)

optimized_model = grid_search.best_estimator_

# =========================================
# 8. VALUTAZIONE FINALE SUL TEST SET
# =========================================

y_pred = optimized_model.predict(X_test)

test_mse = mean_squared_error(y_test, y_pred)

test_r2 = r2_score(y_test, y_pred)

print("Test MSE:", test_mse)

print("Test R2:", test_r2)

# =========================================
# 9. VALORI REALI VS PREDETTI
# =========================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Valori reali")
plt.ylabel("Valori predetti")
plt.title("Valori reali vs predetti")
plt.show()

# =========================================
# 10. LEARNING CURVE
# =========================================

train_sizes, train_scores, validation_scores = learning_curve(
    optimized_model,
    X_train,
    y_train,
    cv=kf,
    scoring="neg_mean_squared_error",
    train_sizes=np.linspace(
        0.1,
        1.0,
        10
    )
)

train_mean = -train_scores.mean(axis=1)

validation_mean = -validation_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training Error"
)
plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Validation Error"
)
plt.xlabel("Training Size")
plt.ylabel("MSE")
plt.title("Learning Curve")
plt.legend()
plt.show()

# =========================================
# 11. PCA
# =========================================

scaler_pca = StandardScaler()

X_train_scaled = scaler_pca.fit_transform(X_train)
X_test_scaled = scaler_pca.transform(X_test)

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Varianza spiegata:", pca.explained_variance_ratio_)
print("Varianza totale:", pca.explained_variance_ratio_.sum())

# =========================================
# 12. RIDGE NELLO SPAZIO PCA
# =========================================

best_alpha = grid_search.best_params_["model__alpha"]

model_pca = Ridge(
    alpha=best_alpha
)

model_pca.fit(
    X_train_pca,
    y_train
)

# =========================================
# 13. CREAZIONE GRIGLIA
# =========================================

x_min = X_train_pca[:, 0].min() - 1
x_max = X_train_pca[:, 0].max() + 1

y_min = X_train_pca[:, 1].min() - 1
y_max = X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(
        x_min,
        x_max,
        50
    ),
    np.linspace(
        y_min,
        y_max,
        50
    )
)

grid = np.c_[
    xx.ravel(),
    yy.ravel()
]

zz = model_pca.predict(
    grid
)

zz = zz.reshape(
    xx.shape
)

# =========================================
# 14. PIANO DI REGRESSIONE 3D
# =========================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
ax.plot_surface(xx, yy, zz, alpha=0.4)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("Target")
ax.set_title("Piano di regressione Ridge nello spazio PCA")
plt.show()