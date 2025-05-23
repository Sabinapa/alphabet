import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

csv_file = "znacilke/znacilke_3x3.csv"
grid_size = csv_file.split("_")[1].replace(".csv", "")
test_size = 0.2
random_state = 42
execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

models = {
    "Logistic Regression (C=0.5)": LogisticRegression(C=0.5, max_iter=300),
    "Logistic Regression (C=1.0)": LogisticRegression(C=1.0, max_iter=300),

    "GaussianNB": GaussianNB(),
    "GaussianNB (var_smoothing=1e-8)": GaussianNB(var_smoothing=1e-8),

    "Decision Tree (max_depth=3)": DecisionTreeClassifier(max_depth=3),
    "Decision Tree (max_depth=7)": DecisionTreeClassifier(max_depth=7),

    "Random Forest (n=10)": RandomForestClassifier(n_estimators=10),
    "Random Forest (n=20)": RandomForestClassifier(n_estimators=20),

    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),

    "AdaBoost (n=20)": AdaBoostClassifier(n_estimators=20),
    "AdaBoost (n=30)": AdaBoostClassifier(n_estimators=30),

    "Gradient Boosting (n=20)": GradientBoostingClassifier(n_estimators=20),
    "Gradient Boosting (n=30)": GradientBoostingClassifier(n_estimators=30),

    "Extra Trees (n=20)": RandomForestClassifier(n_estimators=20, bootstrap=False),
    "Extra Trees (n=40)": RandomForestClassifier(n_estimators=40, bootstrap=False),

    "Passive Aggressive": LogisticRegression(solver="saga", penalty="l2", max_iter=200),
    "Ridge Classifier (C=1.0)": LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=300),
    "Ridge Classifier (C=0.5)": LogisticRegression(penalty="l2", C=0.5, solver="liblinear", max_iter=300),
    "Ridge Classifier (C=2.0)": LogisticRegression(penalty="l2", C=2.0, solver="liblinear", max_iter=300)
}


df = pd.read_csv(csv_file)
X = df.drop(columns=["ime_slike", "tip_crke", "crka", "stevilka"])
y = LabelEncoder().fit_transform(df["crka"])
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


results = []
for name, model in models.items():
    try:
        print(f"Učim model: {name}")
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        end = time.time()
        acc = accuracy_score(y_test, preds)
        duration = round(end - start, 3)
        results.append((grid_size, name, round(acc, 4), duration, test_size, random_state, execution_time))
    except Exception as e:
        results.append((grid_size, name, f"Napaka: {str(e)}", "-", test_size, random_state, execution_time))

results_df = pd.DataFrame(results, columns=[
    "Delitev", "Model", "Accuracy", "Čas (s)", "Test size", "Random state", "Datum"
])
output_csv = f"rezultati_modelov_{grid_size}.csv"
results_df.to_csv(output_csv, index=False)
print(f"Rezultati shranjeni v '{output_csv}'")
