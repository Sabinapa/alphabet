import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")

def get_csv_files(path: str, recursive: bool = True):
    """
    Vrne seznam poti do CSV datotek v dani mapi.
    Če je recursive=True, bo iskal tudi po podmapah.
    """
    if recursive:
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
    else:
        csv_files = [
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(".csv") and os.path.isfile(os.path.join(path, f))
        ]
    return csv_files


# --- Nastavi mapo in možnost rekurzivnega iskanja ---
input_folder = "rezultati"  # ali "rezultati/1 modeli ali rezultati"
recursive_search = True    # ali False za točno določeno mapo ali TRUE za iskanje v celotni mapi

# --- Pridobi CSV datoteke glede na izbiro ---
csv_files = get_csv_files(input_folder, recursive=recursive_search)

# Združi vse rezultate v en DataFrame
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# --- Pretvori čas in točnost v številke ---
df_all["Čas (s)"] = pd.to_numeric(df_all["Čas (s)"], errors="coerce")
df_all["Accuracy"] = pd.to_numeric(df_all["Accuracy"], errors="coerce")

# --- Povprečje točnosti po modelih ---
avg_accuracy = df_all.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
print("\nPovprečna točnost po modelih:")
print(avg_accuracy)

# --- Povprečje po delitvi ---
avg_by_grid = df_all.groupby("Delitev")["Accuracy"].mean().sort_values(ascending=False)
print("\nPovprečna točnost po dimenziji delitve:")
print(avg_by_grid)

# --- Najboljši model po vsaki delitvi ---
best_by_grid = df_all.loc[df_all.groupby("Delitev")["Accuracy"].idxmax()]
print("\nNajboljši modeli po delitvi:")
print(best_by_grid[["Delitev", "Model", "Accuracy", "Čas (s)"]])

# --- Najhitrejši in najpočasnejši modeli ---
fastest = df_all.sort_values(by="Čas (s)").head(5)
slowest = df_all.sort_values(by="Čas (s)", ascending=False).head(5)
print("\nNajhitrejši modeli:", fastest[["Model", "Delitev", "Čas (s)", "Accuracy"]])
print("\nNajpočasnejši modeli:", slowest[["Model", "Delitev", "Čas (s)", "Accuracy"]])

# --- Standardni odklon točnosti po modelih ---
std_by_model = df_all.groupby("Model")["Accuracy"].std().sort_values(ascending=False)
print("\nStandardni odklon točnosti po modelih:")
print(std_by_model)

# --- Povprečni čas izvajanja po modelih ---
avg_time_by_model = df_all.groupby("Model")["Čas (s)"].mean().sort_values()
print("\nPovprečni čas izvajanja po modelih:")
print(avg_time_by_model)

# --- Korelacija med časom in točnostjo ---
correlation = df_all[["Accuracy", "Čas (s)"]].corr()
print("\nKorelacija med časom in točnostjo:")
print(correlation)

# --- Top 5 hitrih in natančnih modelov ---
avg_acc = df_all["Accuracy"].mean()
fast_and_good = df_all[df_all["Accuracy"] > avg_acc].sort_values("Čas (s)").head(5)
print("\nTop 5 hitrih in natančnih modelov:")
print(fast_and_good[["Model", "Delitev", "Accuracy", "Čas (s)"]])

# --- Top modeli po učinkovitosti (Accuracy / Time) ---
df_all["Učinkovitost"] = df_all["Accuracy"] / df_all["Čas (s)"]
top_efficient = df_all.sort_values("Učinkovitost", ascending=False).head(5)
print("\nTop 5 modelov po učinkovitosti (točnost/čas):")
print(top_efficient[["Model", "Delitev", "Accuracy", "Čas (s)", "Učinkovitost"]])

sns.set(style="whitegrid")

# --- 1. Porazdelitev točnosti po dimenziji ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_all, x="Delitev", y="Accuracy", hue="Delitev", palette="pastel", legend=False)
plt.title("Porazdelitev točnosti po dimenziji (Delitev)", fontsize=14)
plt.xlabel("Dimenzija razdelitve slike", fontsize=12)
plt.ylabel("Točnost (Accuracy)", fontsize=12)
plt.tight_layout()
plt.savefig("boxplot_accuracy_by_grid.png")
plt.close()

# --- 2. Porazdelitev časa izvajanja po dimenziji (logaritemska skala) ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_all, x="Delitev", y="Čas (s)", hue="Delitev", palette="muted", legend=False)
plt.title("Porazdelitev časa izvajanja po dimenziji (Delitev)", fontsize=14)
plt.xlabel("Dimenzija razdelitve slike", fontsize=12)
plt.ylabel("Čas izvajanja (v sekundah)", fontsize=12)
plt.yscale("log")  # zaradi velikih razlik v času
plt.tight_layout()
plt.savefig("boxplot_time_by_grid.png")
plt.close()

# --- 3. Povprečna točnost top 10 modelov ---
plt.figure(figsize=(15, 8))
top_models = df_all.groupby("Model")["Accuracy"].mean().sort_values(ascending=False).head(10).index
sns.barplot(data=df_all[df_all["Model"].isin(top_models)], x="Model", y="Accuracy", hue="Model", palette="coolwarm", legend=False)
plt.title("Povprečna točnost top 10 modelov", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Povprečna točnost", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("barplot_top10_models.png")
plt.close()

# --- 4. Povprečna točnost po vseh modelih ---
plt.figure(figsize=(15, 8))
all_models_sorted = df_all.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
sns.barplot(x=all_models_sorted.index, y=all_models_sorted.values, hue=all_models_sorted.index, palette="crest", legend=False)
plt.title("Povprečna točnost po vseh modelih", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Povprečna točnost", fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("barplot_all_models.png")
plt.close()

