"""
Módulo: data_wrangling.py
Responsável por funções de transformação e limpeza de dados após o pré-processamento inicial.

Etapas implementadas:
1. Importação dos datasets de treino e teste.
2. Identificação de colunas numéricas e categóricas.
3. Tratamento de outliers (log-transform e capping).
4. Escalonamento de variáveis numéricas.
5. Codificação de variáveis categóricas (LabelEncoder + OneHotEncoder).
"""

def importando_treino_teste(datasets):
    """
    Importa os datasets de treino e teste.
    Parâmetros:
    : datasets ⇾ nome dos datasets, o nome é padronizado:
    Retorna:
    : Dataframes com as features e valores alvo separados em treino e teste:
    """
    from data_preprocessing import import_dataset
    data = {name: import_dataset(name) for name in datasets}
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    print("Os dataset foram carregados!")
    for name, df in data.items():
        print(f"{name}: {df.shape}")
    return X_train, X_test, y_train, y_test

def identificar_colunas(df):
    """
    Separa as colunas entre numéricas e categóricas.
    Parâmetros:
    : df ⇾ dataset:
    Retorna:
    : Colunas separadas por tipo dos dados:
    """
    numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categoricas = df.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]']).columns.tolist()
    print("\nColunas numéricas:", numericas)
    print("Colunas categóricas:", categoricas,"\n")
    return numericas, categoricas

def tratar_outliers(X_train, X_test, col, metodo):
    """
    Trata outliers em uma coluna específica usando o métod definido.
    Parâmetros:
    :X_train, X_test ⇾ DataFrames de treino e teste:
    :col ⇾ nome da coluna:
    :metodo ⇾ 'capping' ou 'log':
    Retorna:
    :X_train_tratado, X_test_tratado:
    """
    import numpy as np
    X_train_tratado = X_train.copy()
    X_test_tratado = X_test.copy()

    if metodo == 'capping':
        Q1 = X_train_tratado[col].quantile(0.25)
        Q3 = X_train_tratado[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR

        X_train_tratado[col] = np.clip(X_train_tratado[col], limite_inf, limite_sup)
        X_test_tratado[col] = np.clip(X_test_tratado[col], limite_inf, limite_sup)

    elif metodo == 'log':
        X_train_tratado[col] = np.log1p(X_train_tratado[col])
        X_test_tratado[col] = np.log1p(X_test_tratado[col])

    else:
        raise ValueError("Método inválido. use 'capping' or 'log'.")

    print(f"Coluna {col} foi tratada utilizando o metodo {metodo}!\n")
    return X_train_tratado, X_test_tratado

def escalar_valores(X_train, X_test, col_num, metodo = 'standard'):
    """
    Escala as colunas numéricas dos datasets de treino e teste.
    Parâmetros:
    :X_train, X_test ⇾ DataFrames de treino e teste.:
    :col_num ⇾ lista com as colunas numéricas:
    Métodos:
    :'standard' ⇾ StandardScaler (média=0, desvio=1):
    :'minmax'   ⇾ MinMaxScaler (escala entre 0 e 1):
    Retorna:
    :X_train_scaled, X_test_scaled ⇾ DataFrames escalonados:
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if metodo == 'standard':
        scaler = StandardScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Método Invalido. use 'standard' or 'minmax'")

    X_train_scaled[col_num] = scaler.fit_transform(X_train_scaled[col_num])
    X_test_scaled[col_num] = scaler.transform(X_test_scaled[col_num])

    print(f"As colunas numéricas: {col_num}, foram escalonadas utilizando o método:{metodo}\n")
    return X_train_scaled, X_test_scaled

def transformar_colunas_categoricas(X_train, X_test, numericas, categoricas):
    """
    Transforma as variáveis categóricas
    - As colunas em que a ordem importa, como 'loan_grade', utilizam LabelEncoder.
    - As colunas em que os valores não possuem hierarquia utilizam OneHotEncoder.
    Parâmetros:
    : X_train, X_test ⇾ datasets de treino e teste:
    : numericas, categoricas ⇾ separação por tipo dos dados:
    Retorna:
    : Datasets de treino e teste prontos para o uso no modelo:
    """

    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.compose import ColumnTransformer

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    label_encoder = LabelEncoder()
    X_train_encoded['loan_grade'] = label_encoder.fit_transform(X_train_encoded['loan_grade'])
    X_test_encoded['loan_grade'] = label_encoder.transform(X_test_encoded['loan_grade'])

    categoricas_oh = [col for col in categoricas if col != 'loan_grade']

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categoricas_oh),
        ],
        remainder='passthrough'
    )

    X_train_ready = preprocessor.fit_transform(X_train_encoded)
    X_test_ready = preprocessor.transform(X_test_encoded)

    encoded_cols = preprocessor.named_transformers_['categorical'].get_feature_names_out(categoricas_oh)
    todas_cols = list(encoded_cols) + numericas + ['loan_grade']

    print("Formato final dos dados de treino:", X_train_ready.shape)
    print("Formato final dos dados de teste:", X_test_ready.shape)
    print("\nTotal de features após encoding:", len(todas_cols))

    return X_train_ready, X_test_ready, todas_cols