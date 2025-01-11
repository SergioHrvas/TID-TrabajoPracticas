import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar los datos desde el archivo CSV
df = pd.read_csv('./data/heart.csv')

# Mostrar las primeras filas para verificar los datos
print("Datos originales:")
print(df.head())

# Paso 2: Convertir variables categóricas a numéricas
label_encoders = {}
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Paso 3: Verificar si hay valores nulos y tratarlos (si es necesario)
if df.isnull().sum().any():
    print("Hay valores nulos en el dataset, los cuales serán rellenados con la media de la columna.")
    df.fillna(df.mean(), inplace=True)

# Paso 4: Estandarizar las características numéricas
scaler = StandardScaler()
numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Mostrar el DataFrame procesado
print("Datos procesados:")
print(df.head())

# =======================
# Parte 1: Clustering
# =======================
# Selección de características para clustering
features = df.drop('HeartDisease', axis=1)

# Determinar el número óptimo de clústeres usando el método del codo
inertia = []
range_clusters = range(1, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Visualización del método del codo
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para determinar k')
plt.grid()
plt.show()

# Basándonos en el gráfico, seleccionamos un número de clústeres (por ejemplo, k=3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features)

# Reducción de dimensionalidad para visualización
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features)

# Visualización de clústeres
df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2', s=100)
plt.title('Visualización de Clústeres')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Clúster')
plt.grid()
plt.show()

# =========================
# Parte 2: Regresión Logística
# =========================
# Separar datos en características (X) y etiqueta (y)
X = df.drop(['HeartDisease', 'Cluster', 'PCA1', 'PCA2'], axis=1)  # Eliminamos columnas no relevantes
y = df['HeartDisease']

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar el modelo de Regresión Logística
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predecir probabilidades de mortalidad
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluación del modelo
print("\nInforme de clasificación:")
y_pred = (y_pred_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Calcular AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC: {auc_score:.2f}")

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# =========================
# Parte 3: Análisis Conjunto
# =========================
# Incorporar probabilidades y etiquetas predichas al DataFrame
df_test = X_test.copy()
df_test['True_Label'] = y_test
df_test['Predicted_Probability'] = y_pred_prob
df_test['Predicted_Label'] = y_pred

# Analizar perfiles de clústeres en términos de probabilidad de enfermedad
cluster_analysis = df.groupby('Cluster').mean()
print("\nAnálisis de clústeres:")
print(cluster_analysis)

# Mostrar ejemplos de predicciones
print("\nEjemplos de predicciones:")
print(df_test.head())
