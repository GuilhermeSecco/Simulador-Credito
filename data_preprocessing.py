"""
Módulo: data_preprocessing.py
Responsável pelo pré-processamento inicial dos dados antes das etapas de transformação e modelagem.

Etapas implementadas:
1. Importação do dataset bruto.
2. Separação entre features (X) e variável-alvo (y).
3. Divisão dos dados em conjuntos de treino e teste.
4. Identificação e tratamento de valores nulos.
5. Exportação dos datasets processados para a pasta /data.

Autor: Guilherme Fernandes Secco
Versão: 1.0
"""

def import_dataset(df):
    """
    Importa um arquivo CSV da pasta data/ e retorna um DataFrame.
    Parâmetros:
    : df ⇾ nome do arquivo (sem extensão):
    Retorna:
    : Dataframe com os dados:
    """
    import pandas as pd
    return pd.read_csv(f"./data/{df}.csv")

def atribuir_valor_alvo(df, valor_alvo):
    y = df[valor_alvo]
    X = df.drop(columns=[valor_alvo])
    return X, y

def dividir_treino_teste(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def tratar_nulo_com_mediana_treino_teste(X_train, X_test):
    """
    Substitui valores nulos em 'loan_int_rate' pela mediana correspondente ao 'loan_grade',
    calculada apenas no conjunto de treino. Evita data leakage e garante consistência.
    """
    medianas = X_train.groupby('loan_grade')['loan_int_rate'].median()

    for grade, mediana in medianas.items():
        X_train.loc[X_train['loan_grade'] == grade, 'loan_int_rate'] = (
            X_train.loc[X_train['loan_grade'] == grade, 'loan_int_rate'].fillna(mediana)
        )
        X_test.loc[X_test['loan_grade'] == grade, 'loan_int_rate'] = (
            X_test.loc[X_test['loan_grade'] == grade, 'loan_int_rate'].fillna(mediana)
        )

    return X_train, X_test

def export_dataset(df, name, folder='data'):
    """
    Exporta um arquivo CSV para a pasta "data"
    Parâmetros: df ⇾ dataframe:
    : name ⇾ nome do arquivo:
    : folder ⇾ nome da pasta:
    """
    import os
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.csv")
    try:
        df.to_csv(path, index=False)
        print(f"\n{name}.csv exportado com sucesso!")
    except Exception as e:
        print(f"Erro ao exportar o dataset {name}: {e}")

def main():
    """
    Executa o pipeline de pré-processamento:
    1. Importa e limpa os dados brutos.
    2. Atribuir valor alvo.
    3. Divide entre treino e teste.
    4. Trata valores nulos com a mediana.
    5. Exporta os datasets processados.
    """

    #Importando dataset
    df = import_dataset("emprestimos_concebidos")

    #Já sabemos pela EDA que o dataset possui valores nulos
    print(df.isnull().sum(),"\n")

    #Separando o valor alvo do resto
    X, y = atribuir_valor_alvo(df, "loan_status")

    #Separando o dataset entre treino e teste
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)

    #Analisando quantos valores nulos ficaram em cada lado
    print(X_train.isnull().sum(),"\n")
    print(X_test.isnull().sum(),"\n")

    #Aplicando a função de preencher valores nulos com a mediana
    X_train, X_test = tratar_nulo_com_mediana_treino_teste(X_train, X_test)

    #Confirmando se todos os valores nulos foram corrigidos
    print(X_train.isnull().sum(),"\n")
    print(X_test.isnull().sum(),"\n")

    #Exportando os datasets
    export_dataset(X_train, name="X_train")
    export_dataset(X_test, name="X_test")
    export_dataset(y_train, name="y_train")
    export_dataset(y_test, name="y_test")

    print("\n✅ Pré-processamento concluído com sucesso!")
    print(f"\nTreino: {X_train.shape[0]} linhas | Teste: {X_test.shape[0]} linhas")

if __name__ == "__main__":
    main()