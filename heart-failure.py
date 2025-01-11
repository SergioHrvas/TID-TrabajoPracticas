import kaggle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define el nombre del dataset
# dataset_name = 'heart-failure-prediction'  # Cambia a tu dataset
# dataset_owner = 'fedesoriano'  # Cambia según el propietario y el nombre del dataset

# Descargar el dataset
# kaggle.api.dataset_download_files(f'{dataset_owner}/{dataset_name}', path='./data', unzip=True)


# Paso 1: Cargar los datos desde el archivo CSV
df = pd.read_csv('./data/heart.csv')

# Mostrar las primeras filas para verificar los datos
print(df.head())


# Paso 2: Convertir variables categóricas a numéricas
label_encoders = {}

# Codificar columnas categóricas
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

# Seleccionamos las columnas numéricas
numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Mostrar el DataFrame procesado
print(df.head())