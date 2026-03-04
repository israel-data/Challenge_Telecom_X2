"""
-Autor: Israel MArtínez González
-Fecha: Febrero 2026.
-Estado del proyecto: Versión 1.1.0.
-Descripción del Proyecto: Challenge Telecom 2.
-Para ejecutarlo: Python 3.13.11, Libreria Pandas, numpy, matplotlib, seaborn, Scifi-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE


# CARGA Y NORMALIZACIÓN
ruta = "TelecomX_Data.json"
data = pd.read_json(ruta)

df = pd.json_normalize(
    data.to_dict(orient="records"),
    sep="_"
)

# Normalización columnas
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

# Normalizar textos
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.lower().str.strip()

# LIMPIEZA

df.drop_duplicates(inplace=True)
df.replace("", np.nan, inplace=True)

col_contract = [c for c in df.columns if "contract" in c][0]
col_payment = [c for c in df.columns if "payment" in c][0]
col_tenure = [c for c in df.columns if "tenure" in c][0]
col_total = [c for c in df.columns if "total" in c and "charge" in c][0]
col_monthly = [c for c in df.columns if "monthly" in c and "charge" in c][0]

df[col_total] = pd.to_numeric(df[col_total], errors="coerce")
df[col_total].fillna(0, inplace=True)

df.fillna(0, inplace=True)


# VARIABLES BINARIAS

df["churn"] = df["churn"].replace({"yes":1,"no":0}).astype(int)

for col in df.select_dtypes(include="object"):
    if df[col].nunique() == 2:
        df[col] = df[col].replace({"yes":1,"no":0})


# FEATURE ENGINEERING

df["cuenta_diaria"] = df[col_monthly] / 30
df["gasto_promedio_mensual_real"] = df[col_total] / (df[col_tenure] + 1)

df["segmento_tenure"] = pd.cut(
    df[col_tenure],
    bins=[0,12,24,48,72],
    labels=["nuevo","estable","fiel","muy_fiel"]
)

# HEATMAP DE CORRELACIÓN

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Mapa de Calor - Correlación")
plt.show()

# PREPARACIÓN MODELOS
X = df.drop("churn", axis=1)
y = df["churn"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SMOTE BALANCEO
print("Distribución original:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Distribución después de SMOTE:")
print(y_train_res.value_counts())

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# MODELO REGRESIÓN LOGÍSTICA
modelo_log = LogisticRegression(max_iter=2000)
modelo_log.fit(X_train_scaled, y_train_res)

y_pred_log = modelo_log.predict(X_test_scaled)
y_prob_log = modelo_log.predict_proba(X_test_scaled)[:,1]

print("\n REGRESIÓN LOGÍSTICA.")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("AUC:", roc_auc_score(y_test, y_prob_log))
print(classification_report(y_test, y_pred_log))

# MATRIZ DE CONFUSIÓN GRÁFICA
cm = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

# CURVA ROC
fpr, tpr, _ = roc_curve(y_test, y_prob_log)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title("Curva ROC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

# RANDOM FOREST

modelo_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

modelo_rf.fit(X_train_res, y_train_res)

y_pred_rf = modelo_rf.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Importancia variables
importancias = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": modelo_rf.feature_importances_
}).sort_values(by="Importancia", ascending=False)

print("\nTop 10 Variables Importantes:")
print(importancias.head(10))
