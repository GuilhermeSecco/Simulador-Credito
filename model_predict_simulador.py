import joblib
import pandas as pd
import os

def carregar_modelo(caminho_modelo="models/XGBClassifier_simulador.pkl",
                    caminho_colunas="models/feature_columns_simulador.pkl",
                    caminho_preproc="models/preprocessor_simulador.pkl"):
    """
    Carrega o modelo, o pré-processador e as colunas salvas durante o treino.
    """

    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado em: {caminho_modelo}")
    modelo = joblib.load(caminho_modelo)
    print(f"Modelo carregado: {modelo.__class__.__name__}")

    if not os.path.exists(caminho_preproc):
        raise FileNotFoundError(f"Pré-processador não encontrado em: {caminho_preproc}")
    preprocessor = joblib.load(caminho_preproc)
    print(f"Pré-processador carregado com sucesso!")

    if not os.path.exists(caminho_colunas):
        raise FileNotFoundError(f"Arquivo de colunas não encontrado em: {caminho_colunas}")
    feature_cols = joblib.load(caminho_colunas)
    print(f"Colunas carregadas ({len(feature_cols)} features).")

    return modelo, preprocessor, feature_cols

def preprocessar_dados(dados_cliente, preprocessor, feature_cols):
    """
    Recebe os dados de um cliente fictício, aplica o pré-processamento e retorna o DataFrame pronto para o modelo.
    """
    #Convertendo o dicionário em DataFrame
    df_input = pd.DataFrame([dados_cliente])

    print("\nDados brutos recebidos:")
    print(df_input)

    #Aplicando o pré-processador
    X_ready = preprocessor.transform(df_input)

    #Criando o DataFrame com os nomes das colunas originais usadas no treino
    X_ready_df = pd.DataFrame(X_ready, columns=feature_cols)

    print("\nAmostra dos dados processados:")
    print(X_ready_df.head(1))

    return X_ready_df

def prever_risco_credito(modelo, X_ready, threshold=0.5):
    """
    Gera a previsão de inadimplência com base nos dados processados.
    Retorna dicionário com o risco e decisão de crédito.
    """

    # Garante que o modelo suporte predict_proba
    if not hasattr(modelo, "predict_proba"):
        raise ValueError("O modelo não suporta previsão de probabilidade.")

    # Probabilidade de inadimplência (classe 1)
    prob_default = modelo.predict_proba(X_ready)[:, 1][0]

    # Aplica a regra de decisão
    aprova = prob_default < threshold

    # Mensagem humanizada
    resultado = "✅ Crédito Aprovado" if aprova else "❌ Crédito Negado"

    print(f"\nResultado: {resultado} | Risco de inadimplência: {prob_default*100:.2f}%")

    return {
        "resultado": resultado,
        "risco_inadimplencia": round(prob_default, 4),
        "aprova": aprova
    }

if __name__ == '__main__':
    modelo, preproc, cols = carregar_modelo()

    dados_teste = {
        "person_age": 30,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "loan_intent": "PERSONAL",
        "loan_grade": "C",
        "loan_amnt": 8000,
        "loan_int_rate": 12.5,
        "loan_to_income_ratio": 0.16,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }

    X_ready = preprocessar_dados(dados_teste, preproc, cols)
    resultado = prever_risco_credito(modelo, X_ready, threshold=0.5)

    print(resultado)