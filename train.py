import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/manufacturing_defects.csv"
MODEL_DIR = "models"
PLOT_DIR = "plots"

TARGET = "defect"

TEST_SIZE = 0.20
RANDOM_STATE = 42


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df[TARGET].value_counts())


# =========================================================
# DATA PREPARATION
# =========================================================

# Remove rows containing missing values
df = df.dropna()

X = df.drop(columns=[TARGET])
y = df[TARGET]


print("\nNumber of features:", X.shape[1])
print("Number of records:", X.shape[0])


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# DEFINE MODELS
# =========================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# =========================================================
# TRAIN AND EVALUATE
# =========================================================

results = {}

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    trained_models[name] = model

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(
    results
).T

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)


results_df.to_csv(
    "model_comparison.csv"
)


# =========================================================
# ACCURACY COMPARISON PLOT
# =========================================================

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    x=results_df.index,
    y=results_df["Accuracy"]
)

plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Defect Prediction Model Comparison")

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/model_comparison.png",
    dpi=300
)

plt.close()


# =========================================================
# CONFUSION MATRICES
# =========================================================

for name, model in trained_models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "No Defect",
            "Defect"
        ],
        yticklabels=[
            "No Defect",
            "Defect"
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        f"{PLOT_DIR}/{filename}",
        dpi=300
    )

    plt.close()


# =========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

rf = trained_models["Random Forest"]

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance
)


feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)


# =========================================================
# TOP 3 DRIVERS
# =========================================================

top_3 = feature_importance.head(3)

print("\nTOP 3 DEFECT DRIVERS")

for i, row in top_3.iterrows():

    print(
        f"{row['Feature']}: "
        f"{row['Importance']:.4f}"
    )


# =========================================================
# FEATURE IMPORTANCE PLOT
# =========================================================

plt.figure(
    figsize=(10, 7)
)

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.xlabel(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "Manufacturing Parameter"
)

plt.title(
    "Manufacturing Parameter Importance"
)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/feature_importance.png",
    dpi=300
)

plt.close()


# =========================================================
# SAVE RANDOM FOREST MODEL
# =========================================================

joblib.dump(
    rf,
    f"{MODEL_DIR}/random_forest.pkl"
)


# Save feature names
joblib.dump(
    list(X.columns),
    f"{MODEL_DIR}/features.pkl"
)


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Random Forest Accuracy: "
    f"{results['Random Forest']['Accuracy'] * 100:.2f}%"
)

print(
    f"Top defect driver: "
    f"{top_3.iloc[0]['Feature']}"
)
