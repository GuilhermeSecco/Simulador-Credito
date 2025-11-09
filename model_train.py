import pandas as pd
from data_wrangling import *
from plot_utils import salvar_plot
import time

def treinar_modelo(X_train, y_train, biblioteca, model, **config):
    #Importando as bibliotecas
    import importlib
    import numpy as np
    modulo = importlib.import_module(biblioteca)

    #Importando o modelo
    modelo_cls = getattr(modulo, model)

    #Configurando o modelo
    modelo = modelo_cls(**config)

    #Convertendo o y_train para um array 1D
    y_train = np.ravel(y_train)

    #Treinado o modelo
    modelo.fit(X_train, y_train)

    #Mensagem de conclusão
    print(f"\nModelo: {modelo.__class__.__name__}, treinado com sucesso!")
    return modelo

def avaliar_modelo(modelo, X_test, y_test, pasta_plots):
    #Importando as bibliotecas
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report, roc_curve, auc)
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    #Garantindo que o y_test seja um array 1D
    y_test = np.ravel(y_test)

    #Fazendo as previsões
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    #Métricas numéricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #Avaliação do modelo
    print("\n📊 Avaliação do Modelo:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # ==========================
    # 📈 Gráficos e Visualizações
    # ==========================

    # 1️⃣ Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pago", "Não Pago"],
                yticklabels=["Pago", "Não Pago"])
    plt.xlabel("")
    plt.ylabel("")
    plt.title(f"Matriz de Confusão - {modelo.__class__.__name__}")
    salvar_plot(modelo.__class__.__name__, pasta_base=pasta_plots, model=True, metrica="Matriz_Confusao")
    plt.tight_layout()
    plt.close()

    # 2️⃣ Curva ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"Curva ROC - {modelo.__class__.__name__}")
        plt.xlabel("Falsos Positivos")
        plt.ylabel("Verdadeiros Positivos")
        plt.legend()
        salvar_plot(modelo.__class__.__name__, pasta_base=pasta_plots, model=True, metrica="Curva_ROC")
        plt.close()

    # 3️⃣ Importância das features
    if hasattr(modelo, "feature_importances_"):
        importances = modelo.feature_importances_
        #Normaliza automaticamente caso as importâncias não estejam entre 0 e 1
        if importances.max() > 1:
            importances = importances / importances.sum()
        features = X_test.columns
        importancia = pd.Series(importances, index=features).sort_values(ascending=False)

        # Mostrando o top 10  no console
        print(f"\nTop 10 variáveis mais importantes para o modelo: {modelo.__class__.__name__}")
        print(importancia.head(10).round(4).to_string())

        #Salvando em CSV
        import os
        import pandas as pd
        pasta_modelo = os.path.join(pasta_plots, modelo.__class__.__name__)
        os.makedirs(pasta_modelo, exist_ok=True)
        caminho_csv = os.path.join(pasta_modelo, "Importancia_Features.csv")
        importancia.to_csv(caminho_csv, header=["Importância"], index_label="Variável")
        print(f"📁 Importâncias salvas em: {caminho_csv}\n")

        # 🔹 Gráfico de barras horizontais
        top_n = 8
        top_features = importancia.head(top_n)

        # Para exibir a mais importante no topo, invertemos a ordem para o barh
        top_features_rev = top_features[::-1]

        values = top_features_rev.values
        names = top_features_rev.index

        import matplotlib as mpl

        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')

        plt.gca().set_facecolor('#111111')
        plt.gcf().set_facecolor('#111111')

        # Gradiente (cmap) com normalização pelos valores
        cmap = plt.cm.YlGnBu
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        colors = cmap(norm(values))

        y_pos = np.arange(len(top_features_rev))

        bars = plt.barh(y_pos, values, color=colors, edgecolor='black')

        # Monta rótulos com ranking: o item mais importante tem rank 1
        # Como top_features_rev está invertido, o rank é (top_n - i)
        y_labels = [f"{top_n - i} - {name}" for i, name in enumerate(names)]
        plt.yticks(y_pos, y_labels, fontsize=10)

        plt.xlabel("Importância (proporcional)", fontsize=12)
        plt.title(f"Top {top_n} Features - {modelo.__class__.__name__}", fontsize=14)

        # Ajusta limite x para dar espaço aos textos de porcentagem
        xmax = values.max() * 1.12
        plt.xlim(0, xmax)

        # Adiciona valores em porcentagem ao lado direito das barras
        for i, v in enumerate(values):
            pct = v * 100  # converte para percentuais
            plt.text(v + xmax * 0.01, i, f"{pct:.1f}%", va='center', fontsize=12)

        plt.tight_layout()
        salvar_plot(modelo.__class__.__name__, pasta_base=pasta_plots, model=True, metrica="Importancia_Features")
        plt.close()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}

def comparar_modelos(X_train, X_test, y_train, y_test, modelos_dict, pasta_plots):
    resultados = {}
    modelos_treinados = {}

    print("\nIniciando comparativo entre os modelos...\n")
    tempo_total_inicio = time.time()

    for nome, (biblioteca, classe, config) in modelos_dict.items():
        print(f"Treinando modelo: {nome}\n")
        inicio = time.time()

        print("Verificando tamanhos dos datasets:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}\n")

        try:
            modelo = treinar_modelo(X_train, y_train, biblioteca, classe, **config)
            modelos_treinados[nome] = modelo
            metricas = avaliar_modelo(modelo, X_test, y_test, pasta_plots=pasta_plots)
            fim = time.time()

            duracao = fim - inicio
            metricas["tempo_segundos"] = round(duracao, 2)
            resultados[nome] = metricas

            print(f"⏱️ Tempo de execução ({nome}): {duracao:.2f} segundos")

        except Exception as e:
            import traceback
            print(f"\n⚠️ Erro ao treinar {nome}: {e}")
            traceback.print_exc()

    tempo_total_fim = time.time()
    tempo_total = tempo_total_fim - tempo_total_inicio
    print(f"\n🏁 Todos os modelos foram avaliados em {tempo_total:.2f} segundos.")

    resultados_df = pd.DataFrame(resultados).T.sort_values(by="f1_score", ascending=False)
    print("\n📊 Resultados comparativos:\n", resultados_df.round(4))

    melhor_nome = max(resultados, key=lambda m: (resultados[m]["f1_score"] or 0))
    print(f"\n🏆 Melhor modelo com base no F1-Score: {melhor_nome}")

    from plot_utils import plotar_comparativo_roc

    plotar_comparativo_roc(modelos_treinados, X_test, y_test, pasta_plots)

    return resultados_df, modelos_treinados[melhor_nome]

def main(use_smote = False):
    #Lista com os datasets que serão utilizados
    datasets = ["X_train", "X_test", "y_train", "y_test"]

    #Importando os datasets
    X_train, X_test, y_train, y_test = importando_treino_teste(datasets)

    #Corrigindo o formato do Valor-alvo
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    #Separando as colunas em numéricas ou categóricas
    numericas, categoricas = identificar_colunas(X_train)

    #Dicionário com o metodo de tratamento das colunas
    metodos_outliers = {
        'loan_int_rate': 'capping',
        'loan_amnt': 'log',
        'person_income': 'log'
    }

    #Tratamento dos Outliers
    for coluna ,metodo in metodos_outliers.items():
        X_train, X_test = tratar_outliers(X_train, X_test, coluna, metodo)

    #Escalonamento dos valores numéricos
    X_train, X_test = escalar_valores(X_train, X_test, numericas)

    #Transformação das colunas categóricas com OneHotEncoder ou LabelEncoder
    X_train_ready, X_test_ready, todas_cols = transformar_colunas_categoricas(X_train, X_test, numericas, categoricas)

    #Transformando o X_train e o X_test em Dataframes novamente
    X_train_ready = pd.DataFrame(X_train_ready, columns=todas_cols)
    X_test_ready = pd.DataFrame(X_test_ready, columns=todas_cols)


    #Escolha entre original e SMOTE
    # ============================

    if use_smote:
        print("\n🚀 Aplicando SMOTE para balanceamento das classes...")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_train_ready, y_train = smote.fit_resample(X_train_ready, y_train)
        print("Dados balanceados com sucesso!")
        print(f"Distribuição pós-SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

        pasta_plots = "plots/models_plots_smote"
        resultados_path = "plots/model_results_comparison_SMOTE.csv"
    else:

        print("\n⚙️ Usando dados originais (sem balanceamento).")
        pasta_plots = "plots/models_plots"
        resultados_path = "plots/model_results_comparison.csv"

    #Dicionário com as configurações dos modelos de ML
    modelos_dict = {
        "RandomForest": ("sklearn.ensemble", "RandomForestClassifier",
                         {"n_estimators": 200, "max_depth": 10, "random_state": 42, "n_jobs": -1}),

        "LogisticRegression": ("sklearn.linear_model", "LogisticRegression",
                               {"max_iter": 1000, "solver": "lbfgs"}),

        "GradientBoosting": ("sklearn.ensemble", "GradientBoostingClassifier",
                             {"n_estimators": 150, "learning_rate": 0.1, "random_state": 42}),

        "XGBoost": ("xgboost", "XGBClassifier",
                    {"n_estimators": 2000, "learning_rate": 0.1, "max_depth": 6, "random_state": 42, "n_jobs": -1,
                     "tree_method": "hist"}),

        "LightGBM": ("lightgbm", "LGBMClassifier",
                     {"n_estimators": 2000, "learning_rate": 0.1, "max_depth": -1, "random_state": 42, "n_jobs": -1,
                      "device_type": "gpu", "verbose": 0}),

        "SGDClassifier": ("sklearn.linear_model", "SGDClassifier",
                          {"loss": "modified_huber", "penalty": "l2", "alpha": 0.0001,
                           "max_iter": 2000, "tol": 1e-4, "random_state": 42})
    }

    # Treinamento e Avaliação
    # ============================

    resultados_df, melhor_modelo = comparar_modelos(X_train_ready, X_test_ready, y_train, y_test, modelos_dict, pasta_plots=pasta_plots)

    from joblib import dump
    import os

    # Salva o modelo escolhido
    os.makedirs("models", exist_ok=True)
    caminho_modelo = f"models/{melhor_modelo.__class__.__name__}.pkl"

    dump(melhor_modelo, caminho_modelo)
    print(f" Modelo {melhor_modelo} salvo em: {caminho_modelo}")

    # Exporta tabela de comparação para CSV
    resultados_df.to_csv(resultados_path)
    print("\n📁 Resultados comparativos salvos em: plots/model_results_comparison.csv")

if __name__ == "__main__":
    main(use_smote = False)