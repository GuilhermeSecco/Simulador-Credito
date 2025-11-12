import shap
import pandas as pd
import joblib
from data_wrangling import *
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import os

def explicar_modelo(caminho_modelo="models/XGBClassifier.pkl", caminho_colunas="models/feature_columns.pkl"):
    """
    Cria gráficos para mostrar a influência das variáveis no resultado do modelo.
    Parâmetros:
    : caminho_modelo ⇾ Caminho para o melhor modelo:
    : caminho_colunas ⇾ Caminho para as colunas do modelo:
    Retorna:
    : Plot com a influência das variáveis nomeadas:
    """
    #Carregando modelo e as colunas das features
    modelo = joblib.load(caminho_modelo)
    feature_cols = joblib.load(caminho_colunas)

    #Reimportando os datasets de treino e teste
    datasets = ["X_train", "X_test", "y_train", "y_test"]
    X_train, X_test, y_train, y_test = importando_treino_teste(datasets)

    #Separando as colunas Numéricas e Categóricas
    numericas, categoricas = identificar_colunas(X_train)

    #Tratando Outliers
    metodos_outliers = {
        'loan_int_rate': 'capping',
        'loan_amnt': 'log',
        'person_income': 'log'
    }
    for c, m in metodos_outliers.items():
        X_train, X_test = tratar_outliers(X_train, X_test, c, m)

    #Escalando Valores
    X_train, X_test = escalar_valores(X_train, X_test, numericas)
    X_train_ready, X_test_ready, _ = transformar_colunas_categoricas(X_train, X_test, numericas, categoricas)

    #Atribuindo os nomes das colunas aos Dataframes
    X_train_ready = pd.DataFrame(X_train_ready, columns=feature_cols)
    X_test_ready = pd.DataFrame(X_test_ready, columns=feature_cols)

    #Usando o Shap
    explainer = shap.Explainer(modelo.predict, X_train_ready)
    shap_values = explainer(X_test_ready)

    #Gerando Plot de peso das Features
    os.makedirs("plots", exist_ok=True)
    shap.summary_plot(shap_values, X_train_ready, plot_type="bar", show=False)
    plt.title("Importância Global das Features (SHAP Values)")
    plt.tight_layout()
    plt.savefig("plots/shap_importancia_global.png", dpi=300)
    print("📊 Gráfico salvo em: plots/shap_importancia_global.png")
    plt.show()
    plt.close()

    #Criando .CSV com as importâncias médias
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": X_test_ready.columns,
        "Mean(|SHAP|)": feature_importance
    }).sort_values("Mean(|SHAP|)", ascending=False)

    importance_path = "plots/shap_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"📁 Importâncias médias salvas em: {importance_path}")

if __name__ == "__main__":
    explicar_modelo()
