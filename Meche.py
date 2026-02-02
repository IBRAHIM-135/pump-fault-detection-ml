import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


base_path = "/content/drive/MyDrive/pump_data/"


PS1 = pd.read_csv(base_path + "PS1.txt", sep = "\t", header = None)
PS2 = pd.read_csv(base_path + "PS2.txt", sep = "\t", header = None)
PS3 = pd.read_csv(base_path + "PS3.txt", sep = "\t", header = None)
PS4 = pd.read_csv(base_path + "PS4.txt", sep = "\t", header = None)
PS5 = pd.read_csv(base_path + "PS5.txt", sep = "\t", header = None)
PS6 = pd.read_csv(base_path + "PS6.txt", sep = "\t", header = None)

FS1 = pd.read_csv(base_path + "FS1.txt", sep = "\t", header = None)
FS2 = pd.read_csv(base_path + "FS2.txt", sep = "\t", header = None)

EPS = pd.read_csv(base_path + "EPS1.txt", sep = "\t", header = None)

PF = pd.read_csv(base_path + "profile.txt", sep = "\t", header = None)


PS1_mean = PS1.mean(axis=1)
PS2_mean = PS2.mean(axis=1)
PS3_mean = PS3.mean(axis=1)
PS4_mean = PS4.mean(axis=1)
PS5_mean = PS5.mean(axis=1)
PS6_mean = PS6.mean(axis=1)

FS1_mean = FS1.mean(axis=1)
FS2_mean = FS2.mean(axis=1)

EPS_mean = EPS.mean(axis=1)

df = pd.DataFrame({
    "Ps1_mean" : PS1_mean,
    "Ps2_mean" : PS2_mean,
    "Ps3_mean" : PS3_mean,
    "Ps4_mean" : PS4_mean,
    "Ps5_mean" : PS5_mean,
    "Ps6_mean" : PS6_mean,
    "Fs1_mean" : FS1_mean,
    "Fs2_mean" : FS2_mean,
    "Eps_mean" : EPS_mean
})

print("Feature table shape:", df.shape)
print(df.head())

X = df

column = 2
y = PF.iloc[:, column]
print("Label unique values:", y.unique())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


rnd_frst = RandomForestClassifier(n_estimators=200, random_state=42)

voting = VotingClassifier(
    estimators=[
        ("log_reg", log_reg),
        ("dt", dt),
        ("rf", rnd_frst)
    ],
    voting="soft"
)
voting.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_vot = voting.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_vot))

print("\nVoting Classifier Report:")
print(classification_report(y_test, y_pred_vot))

print("\nLogistic Regression Report:")
print(classification_report(y_test, y_pred_log))

print("\nDecision Tree Report:")
print(classification_report(y_test, y_pred_dt))

print("\nConfusion Matrix (Decision Tree):")
c_matrix =confusion_matrix(y_test, y_pred_dt)
print(c_matrix)

print("First 20 predictions:", y_pred_dt[:20])

print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_vot))

print("\nVoting Classifier Report:")
print(classification_report(y_test, y_pred_vot))

proba_vot = voting.predict_proba(X_test)

classes = voting.classes_
print("Classes in voting classifier:", classes)


fault_class = max(classes)
fault_index = list(classes).index(fault_class)

fault_scores = proba_vot[:, fault_index] * 100.0
print("First 20 Fault Probability Scores (%):", fault_scores[:20])


sns.heatmap(c_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix from Decision Tree")
plt.show()

sns.histplot(df["Ps1_mean"], kde=True)
plt.xlabel("PS1_mean (pressure)")
plt.ylabel("Frequency")
plt.title("Histogram for PS1_mean")
plt.show()

plt.scatter(df["Ps1_mean"], df["Fs1_mean"], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("PS1_mean (pressure)")
plt.ylabel("FS1_mean (flowrate)")
plt.title("Pump Signatures: PS1 vs FS1 (colored by condition)")
plt.colorbar(label="Pump condition label")
plt.show()


i = 0

cycle_id = y_test.index[i]

orig_row = df.iloc[cycle_id]

label_pred = y_pred_log[i]
score = fault_scores[i]

print("PUMP HEALTH CERTIFICATE")
print(f"Cycle ID: {cycle_id}")
print(f"PS1 mean (pressure):  {orig_row['Ps1_mean']:.2f}")
print(f"FS1 mean (flowrate):  {orig_row['Fs1_mean']:.2f}")
print(f"EPS1 mean (motor):    {orig_row['Eps_mean']:.2f}")
print("...(other PS/FS features can be added)...")
print(f"Predicted condition label: {label_pred}")
print(f"Fault Probability Score:   {score:.1f}%")
print("Recommendation:",
      "Schedule maintenance." if score > 70 else "Pump OK, continue monitoring.")

lines = [
    "PUMP HEALTH CERTIFICATE",
    "=" * 40,
    f"Cycle ID: {cycle_id}",
    f"PS1 mean (pressure):  {orig_row['Ps1_mean']:.2f}",
    f"FS1 mean (flowrate):  {orig_row['Fs1_mean']:.2f}",
    f"EPS1 mean (motor):    {orig_row['Eps_mean']:.2f}",
    f"Predicted condition label: {label_pred}",
    f"Fault Probability Score:   {score:.1f}%",
    "Recommendation: " + ("Schedule maintenance."
                          if score > 70 else "Pump OK, continue monitoring."),
]

with open("Pump_Health_Certificate.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\nCertificate written to: Pump_Health_Certificate.txt")