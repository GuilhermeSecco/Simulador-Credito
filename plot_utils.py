def salvar_plot(nome: str, pasta_base="plots/Exploratory Data Analysis", timestamp=False, model=False, metrica=False):
    """
    Verifica se o gráfico é de um modelo
    Verifica se o Timestamp é True
    Cria subpastas automaticamente caso necessário.
    Salva o gráfico atual (plt) como imagem PNG dentro da pasta definida.
    Exemplo:
        salvar_plot(modelo.__class__.__name__, model=True, metrica="Matriz_Confusao")
    Isso salvará em:
        plots/models_plots/Nome_do_modelo/Matriz_Confusao.png
    """
    #Bibliotecas necessárias
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    #Atribui a pasta do modelo correspondente caso necessário
    if model:
        pasta_base = pasta_base
        nome_arquivo = f"{nome}/{metrica}.png"
        #Atribui Timestamp se habilitado
        if timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            nome_arquivo = f"{metrica}_{timestamp}.png"
    else:
        #Atribui Timestamp se habilitado
        if timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            nome_arquivo = f"{nome}_{timestamp}.png"
        else:
            nome_arquivo = f"{nome}.png"

    #Atribui o caminho completo
    caminho_completo = os.path.join(pasta_base, nome_arquivo)

    #Garantindo que todas as subpastas existam
    os.makedirs(os.path.dirname(caminho_completo), exist_ok=True)

    #Salvando o gráfico
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo em: {caminho_completo}\n")

def plotar_comparativo_roc(modelos_dict, X_test, y_test, pasta_saida="plots", nome_arquivo="Comparativo_ROC"):
    """
    Cria um gráfico com as curvas ROC de vários modelos no mesmo gráfico e salva em /plots.
    Parâmetros:
        modelos_dict: dicionário {nome_modelo: modelo_treinado}
        X_test, y_test: dados de teste
        pasta_saida: caminho base para salvar o gráfico
        nome_arquivo: nome do arquivo sem extensão
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    import os

    #Lista temporária para armazenar os resultados
    resultados_roc = []

    for nome, modelo in modelos_dict.items():
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
        elif hasattr(modelo, "decision_function"):
            y_proba = modelo.decision_function(X_test)
        else:
            continue  # modelo não tem probabilidade -> ignora

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        resultados_roc.append((nome, fpr, tpr, roc_auc))

    #Ordena pelo valor AUC (decrescente)
    resultados_roc.sort(key=lambda x: x[3], reverse=True)

    #Cria o gráfico
    plt.figure(figsize=(8, 6))
    cores = plt.cm.tab10(np.linspace(0, 1, len(resultados_roc)))

    for (nome, fpr, tpr, roc_auc), cor in zip(resultados_roc, cores):
        plt.plot(fpr, tpr, color=cor, lw=2, label=f"{nome} (AUC = {roc_auc:.3f})")

    #Linha de referência (modelo aleatório)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

    #Título e eixos
    plt.title("Comparativo de Curvas ROC entre Modelos", fontsize=14)
    plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)")
    plt.ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)")

    #Legenda ordenada
    plt.legend(loc="lower right", fontsize=9, title="Modelos (AUC ↓)", title_fontsize=10)
    plt.grid(alpha=0.3)

    #Salvando o gráfico
    os.makedirs(pasta_saida, exist_ok=True)
    caminho = os.path.join(pasta_saida, nome_arquivo + ".png")
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    print(f"📊 Comparativo de Curvas ROC salvo em: {caminho}")

    plt.show()