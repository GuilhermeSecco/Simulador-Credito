import pandas as pd
import joblib
import os

def carregar_modelo(caminho_modelo="models/XGBClassifier.pkl",
                    caminho_colunas="models/feature_columns.pkl",
                    caminho_preproc="models/preprocessor.pkl"):
    """Carrega o modelo treinado, o pré-processador e as colunas."""
    modelo = joblib.load(caminho_modelo)
    print(f"✅ Modelo carregado com sucesso: {modelo.__class__.__name__}")

    if not os.path.exists(caminho_colunas):
        raise FileNotFoundError("❌ Arquivo de colunas (feature_columns.pkl) não encontrado.")
    feature_cols = joblib.load(caminho_colunas)
    print(f"📂 Colunas carregadas ({len(feature_cols)} features).")

    if not os.path.exists(caminho_preproc):
        raise FileNotFoundError("❌ Arquivo do pré-processador (preprocessor.pkl) não encontrado.")
    preprocessor = joblib.load(caminho_preproc)
    print("🧩 Pré-processador carregado com sucesso!")

    return modelo, preprocessor, feature_cols


# ============================
# 🔹 Pré-processamento do input
# ============================

def preprocessar_dados(dados_input: dict, preprocessor, feature_cols):
    """
    Aplica o mesmo pré-processamento usado no treino, garantindo que
    as colunas estejam na mesma ordem e com os mesmos nomes.
    """
    df = pd.DataFrame([dados_input])

    # ✅ Reordena as colunas para coincidir com o pré-processador
    expected_cols = preprocessor.feature_names_in_

    # Adiciona colunas faltantes (com NaN) e reordena corretamente
    df = df.reindex(columns=expected_cols)

    # Log para depuração
    print("\n🧩 Colunas esperadas pelo pré-processador:")
    print(preprocessor.feature_names_in_)

    print("\n📦 Colunas recebidas do input:")
    print(df.columns.tolist())

    # Aplica o pré-processamento salvo
    arr = preprocessor.transform(df)
    df_ready = pd.DataFrame(arr, columns=feature_cols)

    return df_ready


# ============================
# 🔹 Função de predição
# ============================

def prever_cliente(modelo, preprocessor, feature_cols, dados_cliente: dict):
    """Realiza a predição de crédito com segurança e consistência."""
    df_ready = preprocessar_dados(dados_cliente, preprocessor, feature_cols)

    # Garante que o input tenha as mesmas colunas do treino
    for col in feature_cols:
        if col not in df_ready.columns:
            df_ready[col] = 0

    # Remove colunas extras, se houver
    df_ready = df_ready[feature_cols]

    print("\n📊 Amostra dos valores transformados:")
    print(df_ready.head())

    # Faz as previsões
    proba = modelo.predict_proba(df_ready)[0][1]
    pred = modelo.predict(df_ready)[0]

    resultado = "❌ Crédito Negado" if pred == 1 else "✅ Crédito Aprovado"
    print(f"\n🔍 Resultado: {resultado} | Risco de inadimplência: {proba*100:.2f}%")

    return {
        "resultado": resultado,
        "risco_inadimplencia": round(float(proba), 4),
        "aprova": bool(pred == 0)
    }


# ============================
# 🔹 Teste local
# ============================

if __name__ == "__main__":

    modelo, preprocessor, feature_cols = carregar_modelo()

    print("\n🏷️ Classes do modelo:")
    print(modelo.classes_)

    dados_teste = {
        "person_age": 2,
        "person_income": 10,
        "person_emp_length": 5,
        "loan_amnt": 1500000,
        "loan_int_rate": 12.5,
        "loan_percent_income": 10000,
        "cb_person_cred_hist_length": 10,
        "person_home_ownership": "MORTGAGE",
        "loan_intent": "EDUCATION",
        "loan_grade": "D",
        "cb_person_default_on_file": "Y"
    }

    print(f"🔍 Tipo real do modelo carregado: {type(modelo)}")

    import numpy as np

    resultado = prever_cliente(modelo, preprocessor, feature_cols, dados_teste)
    print(resultado)
