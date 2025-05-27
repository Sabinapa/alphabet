import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

modeli1 = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree (depth=5)": DecisionTreeClassifier(max_depth=5),
    "Decision Tree (depth=None)": DecisionTreeClassifier(max_depth=None),
    "Random Forest": RandomForestClassifier(),
    "SVM Linear (C=1)": SVC(kernel="linear", C=1),
    "SVM Linear (C=10)": SVC(kernel="linear", C=10),
    "SVM RBF (gamma=scale)": SVC(kernel="rbf", gamma="scale"),
    "SVM RBF (gamma=auto)": SVC(kernel="rbf", gamma="auto"),
    "Naive Bayes": GaussianNB(),
    "MLP (relu)": MLPClassifier(activation="relu", max_iter=500),
    "MLP (tanh)": MLPClassifier(activation="tanh", max_iter=500),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

modeli2 = {
    # Logistic Regression with different regularization strengths
    "Logistic Regression (C=0.5)": LogisticRegression(C=0.5, max_iter=300),
    "Logistic Regression (C=1.0)": LogisticRegression(C=1.0, max_iter=300),

    # Naive Bayes with standard and custom smoothing
    "GaussianNB": GaussianNB(),
    "GaussianNB (var_smoothing=1e-8)": GaussianNB(var_smoothing=1e-8),

    # Decision Trees with shallow and medium depth
    "Decision Tree (max_depth=3)": DecisionTreeClassifier(max_depth=3),
    "Decision Tree (max_depth=7)": DecisionTreeClassifier(max_depth=7),

    # Random Forest with different number of trees
    "Random Forest (n=10)": RandomForestClassifier(n_estimators=10),
    "Random Forest (n=20)": RandomForestClassifier(n_estimators=20),

    # K-Nearest Neighbors with different k values
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),

    # AdaBoost and Gradient Boosting with varying number of estimators
    "AdaBoost (n=20)": AdaBoostClassifier(n_estimators=20),
    "AdaBoost (n=30)": AdaBoostClassifier(n_estimators=30),
    "Gradient Boosting (n=20)": GradientBoostingClassifier(n_estimators=20),
    "Gradient Boosting (n=30)": GradientBoostingClassifier(n_estimators=30),

    # Simulating Extra Trees with bootstrap=False in Random Forest
    "Extra Trees (n=20)": RandomForestClassifier(n_estimators=20, bootstrap=False),
    "Extra Trees (n=40)": RandomForestClassifier(n_estimators=40, bootstrap=False),

    # Simulated Passive Aggressive via logistic regression with SAGA solver
    "Passive Aggressive": LogisticRegression(solver="saga", penalty="l2", max_iter=200),

    # Ridge Classifier simulated via Logistic Regression with L2 penalty and different C values
    "Ridge Classifier (C=1.0)": LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=300),
    "Ridge Classifier (C=0.5)": LogisticRegression(penalty="l2", C=0.5, solver="liblinear", max_iter=300),
    "Ridge Classifier (C=2.0)": LogisticRegression(penalty="l2", C=2.0, solver="liblinear", max_iter=300)
}

selected_files = [
    "znacilke/znacilke_4x4.csv",
    "znacilke/znacilke_10x10.csv",
    "znacilke/znacilke_15x15.csv"
]

# Define test/train split ratio and seed for reproducibility
test_size = 0.2
random_state = 42
execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
results = []

for csv_file in selected_files:
    grid_size = csv_file.split("_")[1].replace(".csv", "")  # Extract grid size label from file name
    print(f"\nZačenjam obdelavo za: {csv_file}")

    df = pd.read_csv(csv_file)
    X = df.drop(columns=["ime_slike", "tip_crke", "crka", "stevilka"]) # Drop metadata columns
    y = LabelEncoder().fit_transform(df["crka"]) # Encode target labels (letters)

    # Normalize features using standard scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    for name, model in modeli2.items():
        try:
            print(f"Učim model: {name}")
            start = time.time()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            end = time.time()
            acc = accuracy_score(y_test, preds) # Training + prediction duration
            duration = round(end - start, 3)
            results.append((grid_size, name, round(acc, 4), duration, test_size, random_state, execution_time))
        except Exception as e:
            # In case of model failure, record error message
            results.append((grid_size, name, f"Napaka: {str(e)}", "-", test_size, random_state, execution_time))

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results, columns=[
    "Delitev", "Model", "Accuracy", "Čas (s)", "Test size", "Random state", "Datum"
])
results_df.to_csv("rezultati_modelov_vec_dimenzij.csv", index=False)
print("\nRezultati shranjeni v datoteko 'rezultati_modelov_vec_dimenzij.csv'")
