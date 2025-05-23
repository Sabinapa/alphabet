import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

csv_file = "znacilke/znacilke_2x2.csv"
grid_size = csv_file.split("_")[1].replace(".csv", "")
test_size = 0.2
random_state = 42
execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

models = {
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

df = pd.read_csv(csv_file)
X = df.drop(columns=["ime_slike", "tip_crke", "crka", "stevilka"])
y = LabelEncoder().fit_transform(df["crka"])

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

results = []
for name, model in models.items():
    try:
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
print(f"Rezultati shranjeni v datoteko '{output_csv}'")
