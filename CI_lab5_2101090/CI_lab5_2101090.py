# ---------------------------------------------------------------------------
# Lab Assignment 05
# Double Moon Dataset + Linear Classifier + MLNN Classifier
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# ------------------------------------------------------------
# Generate Double Moon Dataset
# ------------------------------------------------------------
def double_moon(n_samples, radius=10, width=6, distance=2):
    """
    Generates the double moon dataset.
    n_samples : number of points per moon
    radius    : main radius of moon
    width     : moon thickness
    distance  : vertical distance between moons
    """

    # First moon
    angles = np.random.uniform(0, np.pi, n_samples)
    r = radius + width * (np.random.rand(n_samples) - 0.5)

    x1 = r * np.cos(angles)
    y1 = r * np.sin(angles)

    # Second moon (shifted)
    x2 = r * np.cos(angles) + radius
    y2 = -r * np.sin(angles) - distance

    X = np.vstack((np.column_stack((x1, y1)),
                   np.column_stack((x2, y2))))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))  # 0 = red, 1 = blue

    return X, y


# -----------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------
if __name__ == "__main__":

    print("Double Moon Dataset Classifier")
    N1 = int(input("Enter number of red class samples (N1): "))
    N2 = int(input("Enter number of blue class samples (N2): "))

    # Generate dataset
    X, y = double_moon(min(N1, N2))  # balanced dataset

    # Train / Val / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Normalize the data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # --------------------------------------------------------------
    # 1) Linear Classifier (Logistic Regression)
    # --------------------------------------------------------------
    lin_model = LogisticRegression()
    lin_model.fit(X_train_s, y_train)

    # Plot Decision Boundary (Linear)
    def plot_linear_boundary(model, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))

        Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10)
        plt.title(title)
        plt.show()

    plot_linear_boundary(lin_model, "Linear Classifier Decision Boundary")

    print("\n=== LINEAR CLASSIFIER RESULTS ===")
    print("Training Accuracy:", lin_model.score(X_train_s, y_train))
    print("Validation Accuracy:", lin_model.score(X_val_s, y_val))

    # --------------------------------------------------------------
    # 2) Multilayer Neural Network (MLNN)
    # --------------------------------------------------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=0
    )

    mlp.fit(X_train_s, y_train)

    # Plot Decision Boundary (MLNN)
    def plot_mlnn_boundary(model, title):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))

        Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10)
        plt.title(title)
        plt.show()

    plot_mlnn_boundary(mlp, "MLNN Classifier Decision Boundary")

    print("\n=== MLNN CLASSIFIER RESULTS ===")
    print("Training Accuracy:", mlp.score(X_train_s, y_train))
    print("Validation Accuracy:", mlp.score(X_val_s, y_val))

    # --------------------------------------------------------------
    # Loss curves (MLNN)
    # --------------------------------------------------------------
    plt.plot(mlp.loss_curve_)
    plt.title("MLNN Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    # --------------------------------------------------------------
    # Comparison Summary
    # --------------------------------------------------------------
    print("\n==================== COMPARISON ====================")
    print("Linear classifier struggles because the dataset is not linearly separable.")
    print("MLNN (nonlinear activation) successfully learns curved decision boundary.")
    print("====================================================\n")

