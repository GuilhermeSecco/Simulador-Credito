# 💳 Simulador de Crédito Inteligente

> Um simulador interativo de aprovação de crédito com **Flask**, **XGBoost** e **Machine Learning**.

Este projeto faz parte do meu portfólio de Data Science e Machine Learning.  
O sistema utiliza um modelo **XGBoost Classifier** treinado para prever o risco de inadimplência com base em informações financeiras e demográficas do usuário.

---

## 🚀 Funcionalidades

- 🧠 **Predição automática** de aprovação de crédito com base no perfil do cliente  
- 💰 **Cálculo da taxa de juros** conforme o grau de crédito (A → G)  
- 📉 **Exibição da probabilidade de inadimplência** estimada pelo modelo  
- 📊 **Gráfico interativo** dos fatores mais influentes na decisão  
- 💬 **Explicação textual** dos critérios e variáveis  
- 🎨 **Interface moderna e responsiva** com Bootstrap e animações CSS  
- 🔄 **Campos persistentes** após simulação (não apagam após envio)

---

## ⚙️ Como funciona

O usuário preenche um formulário com:
- Idade  
- Score de crédito (0–1000)  
- Renda mensal  
- Valor do empréstimo  
- Tipo de residência  
- Finalidade do empréstimo  
- Histórico de inadimplência  

Esses dados passam por um pipeline de pré-processamento e são enviados para o modelo de Machine Learning, que retorna:

- ✅ Aprovação ou rejeição do crédito  
- 📉 Risco estimado de inadimplência  
- 💸 Renda anual estimada  
- 🧾 Score informado e **grau de crédito (A–G)**  
- 💰 Taxa de juros aplicada automaticamente conforme o grau  

---

## 🧩 Modelagem e Treinamento

O modelo foi treinado em dados históricos de solicitações de crédito, com tratamento completo das variáveis:

| Etapa | Descrição |
|-------|------------|
| 🔹 Remoção de `person_emp_length` | Coluna pouco relevante e com muitos nulos |
| 🔹 Criação de `loan_to_income_ratio` | Substitui `loan_percent_income` com métrica mais estável |
| 🔹 Preenchimento de juros faltantes | Usa a mediana das taxas por `loan_grade` |
| 🔹 Balanceamento de classes | Ajuste automático via `scale_pos_weight` |

### 📈 Taxas medianas por grau de crédito:
| Grau | Taxa (%) |
|------|-----------|
| A | 7.49 |
| B | 10.99 |
| C | 13.48 |
| D | 15.31 |
| E | 16.82 |
| F | 18.53 |
| G | 20.11 |

---

## 🤖 Modelo de Machine Learning

O modelo final foi o **XGBoost Classifier**, configurado para equilibrar performance e estabilidade.

```python
XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=10,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    scale_pos_weight=ratio,
    eval_metric="aucpr"
)
```

🧾 Métricas de desempenho:

    Acurácia: ≈ 93%
    
    F1-Score: ~0.82
    
    AUC (ROC): 0.95

🌐 Estrutura do Projeto

    ml_models/
    ├── model_train_simulador.py          # Treinamento do modelo
    ├── model_predict_simulador.py        # Funções de predição e explicação
    ├── preprocessor_simulador.pkl        # Pré-processador salvo
    ├── feature_columns_simulador.pkl     # Colunas usadas pelo modelo
    ├── XGBClassifier_simulador.pkl       # Modelo final
    └── taxas_por_grade.pkl               # Tabela de juros medianos
    
    templates/
    └── projetos/
        └── simulador-credito.html        # Interface Flask
    
    static/
    ├── css/simulador_credito.css         # Estilos específicos
    └── img/projects/simulador credito.png # Imagem de demonstração

## 🧠 Tecnologias Utilizadas
|Categoria|Tecnologias|
|:---|---:|
|Linguagem|Python 3|
|Machine Learning|XGBoost, scikit-learn, pandas, NumPy|
|Web Framework|Flask|
|Frontend|HTML5, Bootstrap 5, Chart.js, Jinja2|
|Outros|Joblib, Animate.css|

## 🧭 Estrutura Lógica do Simulador

O usuário envia os dados via formulário (Flask recebe via POST).

O pré-processador transforma e codifica os dados.

O modelo XGBoost gera a probabilidade de inadimplência.

O Flask calcula o grau de crédito e taxa correspondente.

O resultado é renderizado na interface com explicações e gráficos.

## 🖥️ Demonstração
### 👉 [Acessar Simulador de Crédito](https://portifolio-guilhermesecco.onrender.com/projetos/simulador-credito)


## 🧑‍💻 Autor

### Guilherme Fernandes Secco

### [💼LinkedIn](https://www.linkedin.com/in/guilherme-f-secco/)

### [💻GitHub](https://github.com/GuilhermeSecco)
