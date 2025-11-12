from data_wrangling import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib
import os
import pandas as pd


# Importando os datasets
datasets = ["X_train", "X_test", "y_train", "y_test"]
X_train, X_test, y_train, y_test = importando_treino_teste(datasets)

#Removendo a Coluna person_emp_length
if "person_emp_length" in X_train.columns:
    X_train = X_train.drop(columns=["person_emp_length"])
if "person_emp_length" in X_test.columns:
    X_test = X_test.drop(columns=["person_emp_length"])
print("Coluna 'person_emp_length' removida com sucesso!")
print(f"Novas dimensões -> X_train: {X_train.shape}, X_test: {X_test.shape}")

#Verificando se ainda existem colunas com valores nulos
print(X_train.isnull().sum())

#Calculando as taxas médias de juros por grau de crédito
taxas_por_grade = (
    X_train.groupby("loan_grade")["loan_int_rate"]
    .median()  # ou .mean()
    .round(2)
    .to_dict()
)

print("📈 Taxas medianas por grade:")
for grade, taxa in taxas_por_grade.items():
    print(f"  {grade}: {taxa}%")

# Substitui valores ausentes ou inconsistentes
X_train["loan_int_rate"] = X_train.apply(
    lambda row: taxas_por_grade.get(row["loan_grade"], 12.0)
    if pd.isna(row["loan_int_rate"]) or row["loan_int_rate"] <= 0
    else row["loan_int_rate"],
    axis=1
)

# Aplica o mesmo ajuste para o conjunto de teste
X_test["loan_int_rate"] = X_test.apply(
    lambda row: taxas_por_grade.get(row["loan_grade"], 12.0)
    if pd.isna(row["loan_int_rate"]) or row["loan_int_rate"] <= 0
    else row["loan_int_rate"],
    axis=1
)

# Salva o dicionário de taxas para uso posterior no simulador
os.makedirs("models", exist_ok=True)
joblib.dump(taxas_por_grade, "models/taxas_por_grade.pkl")
print("Tabela de taxas medianas salva em: models/taxas_por_grade.pkl")

#Criando a coluna loan_to_income_ratio
X_train["loan_to_income_ratio"] = X_train["loan_amnt"] / (X_train["person_income"] + 1)
X_test["loan_to_income_ratio"] = X_test["loan_amnt"] / (X_test["person_income"] + 1)

#Substituindo a coluna loan_percent_income pela coluna loan_to_income_ratio
if "loan_percent_income" in X_train.columns:
    X_train = X_train.drop(columns=["loan_percent_income"])
    X_test = X_test.drop(columns=["loan_percent_income"])
    print("Substituindo 'loan_percent_income' por 'loan_to_income_ratio'.")

#identificando colunas
numericas, categoricas = identificar_colunas(X_train)

print("\n📊 Colunas numéricas:", numericas)
print("📁 Colunas categóricas:", categoricas)

#Montando ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", "passthrough", numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas)
], remainder="drop")

#Aplicando transformações
X_train_arr = preprocessor.fit_transform(X_train)
X_test_arr  = preprocessor.transform(X_test)

#Recuperando os nomes das colunas
num_cols = list(numericas)
cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categoricas)
todas_cols = num_cols + list(cat_cols)

#Montando os DataFrames pro treino
X_train_ready = pd.DataFrame(X_train_arr, columns=todas_cols, index=X_train.index)
X_test_ready  = pd.DataFrame(X_test_arr,  columns=todas_cols, index=X_test.index)

print(f"\nPré-processamento finalizado.\nX_train_ready: {X_train_ready.shape}, X_test_ready: {X_test_ready.shape}")
print("Exemplo:")
print(X_train_ready.head())

#Salvando preprocessor e colunas para uso no predict
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/preprocessor_simulador.pkl")
joblib.dump(todas_cols, "models/feature_columns_simulador.pkl")
print("\nPreprocessor salvo em models/preprocessor_simulador.pkl")
print("Lista de features salva em models/feature_columns_simulador.pkl")

#Cálculo automático do peso da classe minoritária
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

#Inicialização do modelo
xgb_model = XGBClassifier(
    n_estimators=2000,                 #Número total de árvores. Valores maiores reduzem o bias, mas podem gerar overfitting.
    learning_rate=0.08,                #Taxa de aprendizado do modelo. Valores maiores resultam em aprendizado mais rápido, porém geram overfitting
    max_depth=6,                       #Profundidade máxima de cada árvore. Valores maiores reduzem o bias, mas podem gerar overfitting.
    subsample=0.9,                     #Age como regularização, ajuda a reduzir a correlação entre árvores para reduzir overfitting, valores comuns: 0.6~0.9
    colsample_bytree=0.8,              #Controla o número de variáveis consideradas a cada árvore. Evita que algumas features dominem todas as divisões. Valores típicos: 0.7~0.9
    random_state=42,                   #Random State fixo para reprodução futura dos mesmos resultados
    n_jobs=-1,                         #n_jobs=-1: usa todos os núcleos da CPU — acelera o treino.
    tree_method="hist",                #Metod interno de construção das árvores. 'auto': escolha automática. 'hist': usa histogramas — mais rápido e consome menos memória. 'gpu_hist': versão otimizada para GPU.
    scale_pos_weight=scale_pos_weight, #Corrige a influência da classe minoritária na função de perda. Melhora recall da classe minoritária.
    base_score=0.5,                    #Define o “chute inicial” do modelo.
    eval_metric="aucpr"                #Métrica usada internamente para avaliar o erro durante o treino. 'logloss' → perda logística (default para classificação binária). 'auc' → área sob a curva ROC. 'error' → taxa de erro simples
)

#Treinando o Modelo
print("\nTreinando modelo XGBoost...")
xgb_model.fit(X_train_ready, y_train)
print("Modelo treinado com sucesso!")

#Salvando o Modelo
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/XGBClassifier_simulador.pkl")
print("Modelo salvo em: models/XGBClassifier_simulador.pkl")