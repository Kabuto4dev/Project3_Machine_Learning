import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# Función para cargar datos
def load_data():
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Análisis de los datos del df
def dataframe_analysis(df):
    # Verificar tipos de datos
    tipos_datos = df.dtypes
    tipos_datos_df = tipos_datos.reset_index()
    tipos_datos_df.columns = ['Column', 'DataType']
    
    # Contar valores faltantes
    valores_faltantes = df.isnull().sum()
    valores_faltantes_df = valores_faltantes.reset_index()
    valores_faltantes_df.columns = ['Column', 'MissingValues']
    
    return tipos_datos_df, valores_faltantes_df

# Limpieza y exploración
def clean_and_explore_data(df):
    # Limpiar datos eliminando filas con valores nulos
    df_clean = df.dropna()

    # Convertir tipos de datos si es necesario
    for column in df_clean.columns:
        if df_clean[column].dtype == 'object':
            df_clean[column] = df_clean[column].astype('category')

    # Filtrar solo las columnas numéricas para el resumen estadístico
    numeric_columns = df_clean.select_dtypes(include=[np.number])

    # Resumen estadístico solo para columnas numéricas
    resumen_estadistico_df = numeric_columns.describe().reset_index()

    return df_clean, resumen_estadistico_df

# Función para mostrar la explicación del modelo
def show_model_explanation(model_option):
    explanations = {
        "KNN": "K-Nearest Neighbors: Este modelo clasifica los datos en función de la proximidad a los puntos de entrenamiento más cercanos.",
        "Regresión Logística": "Regresión Logística: Este modelo se utiliza para predecir la probabilidad de una clase binaria, útil en problemas de clasificación.",
        "Árbol de decisión": "Árbol de Decisión: Este modelo utiliza una estructura de árbol para tomar decisiones basadas en características de los datos.",
        "Bagging": "Bagging: Este modelo mejora la precisión de otros modelos combinando múltiples versiones del mismo modelo entrenadas en diferentes subconjuntos de datos.",
        "Random Forest": "Random Forest: Este modelo utiliza múltiples árboles de decisión entrenados en diferentes partes del mismo conjunto de datos para mejorar la precisión.",
        "AdaBoost": "AdaBoost: Este modelo mejora la precisión de otros modelos ajustando iterativamente los errores de los modelos anteriores.",
        "Gradient Boosting": "Gradient Boosting: Este modelo construye un modelo aditivo de manera iterativa, optimizando la función de pérdida en cada paso.",
        "Regresión Lineal": "Regresión Lineal: Este modelo se utiliza para predecir un valor continuo basándose en la relación lineal entre las características de los datos."
    }
    st.markdown(f"<div style='color: green; font-style: italic;'>{explanations.get(model_option, 'Selecciona un modelo para ver la explicación.')}</div>", unsafe_allow_html=True)

# Elección del modelo ML
def choose_model(target_type, n_neighbors=3):
    model_option = None
    params = {}
    if target_type == 'categorical':
        model_option = st.radio('Elige un modelo:', [
            'KNN', 'Regresión Logística', 'Árbol de decisión', 'Bagging', 'Random Forest', 'AdaBoost', 'Gradient Boosting'
        ])
        show_model_explanation(model_option)
        if model_option == "KNN":
            n_neighbors = st.number_input('Introduce el número de vecinos:', min_value=1, max_value=100, value=n_neighbors)
            model = KNeighborsClassifier()
            params = {'n_neighbors': range(1, 31)}
        elif model_option == "Regresión Logística":
            model = LogisticRegression()
            params = {'C': np.logspace(-4, 4, 20), 'solver': ['liblinear']}
        elif model_option == "Árbol de decisión":
            max_depth = st.slider('Profundidad máxima del árbol:', 1, 20, 5)
            min_samples_leaf = st.slider('Número mínimo de muestras por hoja:', 1, 20, 1)
            max_leaf_nodes = st.slider('Número máximo de hojas:', 2, 50, 10)
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
            params = {'max_depth': range(1, 21), 'min_samples_leaf': range(1, 21), 'max_leaf_nodes': range(2, 51)}
        elif model_option == "Bagging":
            base_model = DecisionTreeClassifier()
            n_estimators = st.number_input('Número de estimadores:', min_value=10, max_value=100, value=50)
            model = BaggingClassifier(base_estimator=base_model)
            params = {'n_estimators': range(10, 101, 10)}
        elif model_option == "Random Forest":
            n_estimators = st.number_input('Número de estimadores:', min_value=10, max_value=100, value=50)
            max_depth = st.slider('Profundidad máxima del árbol:', 1, 20, 5)
            model = RandomForestClassifier()
            params = {'n_estimators': range(10, 101, 10), 'max_depth': range(1, 21)}
        elif model_option == "AdaBoost":
            base_model = DecisionTreeClassifier()
            n_estimators = st.number_input('Número de estimadores:', min_value=10, max_value=100, value=50)
            model = AdaBoostClassifier(base_estimator=base_model)
            params = {'n_estimators': range(10, 101, 10)}
        else:  # Gradient Boosting
            n_estimators = st.number_input('Número de estimadores:', min_value=10, max_value=100, value=50)
            learning_rate = st.slider('Tasa de aprendizaje:', 0.01, 1.0, 0.1)
            model = GradientBoostingClassifier()
            params = {'n_estimators': range(10, 101, 10), 'learning_rate': np.linspace(0.01, 1, 10)}
    else:
        model_option = st.radio('Elige un modelo:', [
            'KNN', 'Regresión Lineal', 'Árbol de decisión', 'Bagging', 'Random Forest', 'AdaBoost', 'Gradient Boosting'
        ])
        show_model_explanation(model_option)
        if model_option == "KNN":
            n_neighbors = st.number_input('Introduce el número de vecinos:', min_value=1, max_value=100, value=n_neighbors)
            model = KNeighborsRegressor()
            params = {'n_neighbors': range(1, 31)}
        elif model_option == "Regresión Lineal":
            model = LinearRegression()
            params = {}
        elif model_option == "Árbol de decisión":
            max_depth = st.slider('Profundidad máxima del árbol:', 1, 20, 5)
            min_samples_leaf = st.slider('Número mínimo de muestras por hoja:', 1, 20, 1)
            max_leaf_nodes = st.slider('Número máximo de hojas:', 2, 50, 10)
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
            params = {'max_depth': range(1, 21), 'min_samples_leaf': range(1, 21), 'max_leaf_nodes': range(2, 51)}
        elif model_option == "Bagging":
            base_model = DecisionTreeRegressor()
            n_estimators = st.number_input('Número de estimadores:', min_value=10, max_value=100, value=50)
            model = BaggingRegressor(base_estimator=base_model)
            params = {'n_estimators': range(10, 101, 10)}
        elif model_option == "Random Forest":
            n_estimators = st.number_input('Número de estimadores:', min_value=10, max_value=100, value=50)
            max_depth = st.slider('Profundidad máxima del árbol:', 1, 20, 5)
            model = RandomForestRegressor()
            params = {'n_estimators': range(10, 101, 10), 'max_depth': range(1, 21)}
        elif model_option == "AdaBoost":
            base_model = DecisionTreeRegressor()
            n_estimators = st.number_input('Número de estimadores:', min_value=10, max_value=100, value=50)
            learning_rate = st.slider('Tasa de aprendizaje:', 0.01, 1.0, 0.1)
            model = AdaBoostRegressor(base_estimator=base_model)
            params = {'n_estimators': range(10, 101, 10), 'learning_rate': np.linspace(0.01, 1, 10)}
        else:  # Gradient Boosting
            n_estimators = st.number_input('Número de estimadores:', min_value=10, max_value=100, value=50)
            learning_rate = st.slider('Tasa de aprendizaje:', 0.01, 1.0, 0.1)
            model = GradientBoostingRegressor()
            params = {'n_estimators': range(10, 101, 10), 'learning_rate': np.linspace(0.01, 1, 10)}
    
    return model, model_option, params

# Función para entrenar y evaluar el modelo
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_type, model_option=None, search_type=None, params=None):
    best_params = None
    if search_type:
        if search_type == 'Grid Search':
            search = GridSearchCV(model, params, cv=5, n_jobs=-1)
        elif search_type == 'Random Search':
            search = RandomizedSearchCV(model, params, n_iter=20, cv=5, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        st.write(f"Mejores hiperparámetros encontrados: {best_params}")
    else:
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    if target_type == 'categorical':
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        return precision, recall, cm, cr, y_pred, best_params
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, rmse, mae, r2, y_pred, best_params

# Función para comprobar y mostrar el balanceo de clases
def check_balance(df, target):
    value_counts = df[target].value_counts()
    st.write("Distribución de la columna objetivo:")
    st.bar_chart(value_counts)
    return value_counts

def main():
    st.title("Sistema de Recomendación con Modelos de Machine Learning")
    
    # Incluir estilo CSS personalizado
    st.markdown(
        """
        <style>
        .optimum-values {
            text-align: right; 
            font-size: 8px; 
            border-top: 1px dotted gray; 
            font-style: italic; 
            color: gray;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Cargar los datos
    data = load_data()
    
    if data is not None:
        st.write("DataFrame original:")
        st.dataframe(data.head())
        
        # Análisis de los datos
        tipos_datos_df, valores_faltantes_df = dataframe_analysis(data)
        st.write("Tipos de Datos:")
        st.dataframe(tipos_datos_df)
        
        st.write("Valores Faltantes:")
        st.dataframe(valores_faltantes_df)
        
        # Limpieza y exploración
        df_clean, resumen_estadistico_df = clean_and_explore_data(data)
        st.write("DataFrame limpio (sin valores nulos):")
        st.dataframe(df_clean.head())
        
        st.write("Resumen Estadístico (solo columnas numéricas):")
        st.dataframe(resumen_estadistico_df)
        
        # Colocar los selectores en el lado izquierdo
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write("Seleccione las columnas")
            target = st.selectbox('Selecciona la columna objetivo:', df_clean.columns)
            
            # Selección de columnas de características relevantes
            features = st.multiselect(
                'Selecciona las columnas de características:',
                df_clean.columns
            )
            
            if target and features:
                # Determinar el tipo de objetivo (categórico o continuo)
                if pd.api.types.is_numeric_dtype(df_clean[target]):
                    target_type = 'continuous'
                else:
                    target_type = 'categorical'

                # Comprobar el balanceo de clases antes de la selección del modelo
                value_counts = check_balance(df_clean, target)
                if len(value_counts) == 2:
                    st.markdown(
                        "<button style='color: red;'>Es aconsejable realizar técnicas de balanceo</button>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<button style='color: green;'>No es necesario realizar técnicas de balanceo</button>",
                        unsafe_allow_html=True
                    )
            
                st.write("Seleccione un modelo")
                model, model_option, params = choose_model(target_type)
                
                st.write("Seleccione el tipo de búsqueda de hiperparámetros")
                search_type = st.radio(
                    "Elija un tipo de búsqueda de hiperparámetros",
                    ('No buscar', 'Grid Search', 'Random Search')
                )
        
        with col2:
            if 'model_option' in locals():
                st.markdown("<h3 style='text-align: center;'><strong>Análisis de resultados</strong></h3>", unsafe_allow_html=True)
                st.write(f"Has seleccionado el modelo: {model_option}")
            
            if target and features:
                # Reducción de datos para pruebas más manejables
                df_clean = df_clean.sample(frac=0.05, random_state=42)  # Reducimos al 5% de los datos
                st.write(f"Número de filas después de la reducción: {df_clean.shape[0]}")  # Imprimir el número de filas

                # Mostrar la media y la desviación estándar de la columna objetivo
                mean_target = df_clean[target].mean()
                std_target = df_clean[target].std()
                st.write(f'Media de la columna objetivo: {mean_target}')
                st.write(f'Desviación Estándar de la columna objetivo: {std_target}')

                X = df_clean[features]
                y = df_clean[target]
                
                # Convertir variables categóricas a dummies
                X = pd.get_dummies(X, drop_first=True)
                
                # Escalar las variables numéricas
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                
                # Dividir los datos en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Entrenar y evaluar el modelo
                if search_type == 'No buscar':
                    search_type = None
                
                if target_type == 'categorical':
                    precision, recall, cm, cr, y_pred, best_params = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_type, model_option, search_type, params)
                    st.write("Matriz de confusión:")
                    st.write(cm)
                    st.write("Reporte de clasificación:")
                    st.text(cr)
                    st.write(f'Precisión: {precision}')
                    st.write(f'Recall: {recall}')
                    # Visualización de la matriz de confusión
                    fig, ax = plt.subplots()
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                    disp.plot(ax=ax)
                    st.pyplot(fig)
                else:
                    mse, rmse, mae, r2, y_pred, best_params = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_type, model_option, search_type, params)
                    st.write(f'Error Cuadrático Medio (MSE): {mse}')
                    st.write(f'Raíz del Error Cuadrático Medio (RMSE): {rmse}')
                    st.write(f'Error Absoluto Medio (MAE): {mae}')
                    st.write(f'R2 Score: {r2}')
                    
                    # Mostrar valores óptimos objetivo
                    st.markdown(
                        """
                        <style>
                        .optimum-values {
                            text-align: right; 
                            font-size: 8px; 
                            border-top: 1px dotted gray; 
                            font-style: italic; 
                            color: gray;
                        }
                        </style>
                        <div class='optimum-values'>
                            <p><strong>Valores Óptimos Objetivo</strong></p>
                            <p>MSE: Lo más cercano a 0</p>
                            <p>RMSE: Lo más cercano a 0</p>
                            <p>MAE: Lo más cercano a 0</p>
                            <p>R2 Score: Lo más cercano a 1</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    # Visualización de resultados
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
                    ax.set_xlabel('Valor Real')
                    ax.set_ylabel('Predicción')
                    st.pyplot(fig)

                    # Gráfico adicional: Residuales
                    residuals = y_test - y_pred
                    fig, ax = plt.subplots()
                    ax.scatter(y_pred, residuals)
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_xlabel('Predicción')
                    ax.set_ylabel('Residual')
                    st.write("Gráfico de Residuales (*dif. entre v.reales y predichos en un modelo de regresión*):")
                    st.pyplot(fig)

                # Mostrar el gráfico del árbol de decisión si el modelo seleccionado es un árbol de decisión
                if model_option == "Árbol de decisión":
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(model, filled=True, ax=ax, feature_names=X.columns, fontsize=10)
                    st.pyplot(fig)

    else:
        st.write("Por favor, carga un archivo CSV para continuar.")

if __name__ == "__main__":
    main()

