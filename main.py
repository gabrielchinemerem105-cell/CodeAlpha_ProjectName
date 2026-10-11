import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "Iris.csv"


def load_dataset():
    """Load the iris dataset from a CSV if available; otherwise use sklearn's builtin dataset."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        required_columns = {
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
            "Species",
        }

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV file is missing required columns: {sorted(missing)}. "
                "Expected columns: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species"
            )

        features = [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ]
        X = df[features]
        y = df["Species"].str.replace("Iris-", "", regex=False)
        return X, y, features

    print(f"'{DATASET_PATH}' not found. Falling back to the built-in Iris dataset.")
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target_names[data.target])
    features = list(X.columns)
    return X, y, features


def main():
    X, y, features = load_dataset()

    print("First five rows:")
    print(X.head())
    print("\nDataset shape:", X.shape)
    print("\nSpecies distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    labels = model.named_steps["kneighborsclassifier"].classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Iris Species Confusion Matrix")
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.tight_layout()
    plt.savefig("iris_confusion_matrix.png")
    plt.show()

    new_flower = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=features,
    )
    prediction = model.predict(new_flower)

    print("\nNew flower measurements:", new_flower.values[0])
    print("Predicted species:", prediction[0])


if __name__ == "__main__":
    main()
