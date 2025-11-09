def comparar_resultados_smote():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Caminhos dos arquivos
    path_normal = "plots/model_results_comparison.csv"
    path_smote = "plots/model_results_comparison_SMOTE.csv"

    # Leitura dos resultados
    df_normal = pd.read_csv(path_normal, index_col=0)
    df_smote = pd.read_csv(path_smote, index_col=0)

    # Garantindo mesma ordem dos modelos
    modelos = df_normal.index
    df_smote = df_smote.reindex(modelos)

    # Criando pasta para salvar os gráficos
    os.makedirs("plots/comparativos", exist_ok=True)

    # Função auxiliar para plotar gráfico comparativo
    def plot_comparativo(metrica, titulo):
        x = np.arange(len(modelos))
        largura = 0.35

        # Converter valores para porcentagem
        normal_pct = df_normal[metrica] * 100
        smote_pct = df_smote[metrica] * 100

        # Calcular variação percentual
        delta_pct = smote_pct - normal_pct

        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(x - largura/2, normal_pct, largura, label="Sem SMOTE", color="#007acc")
        bars2 = plt.bar(x + largura/2, smote_pct, largura, label="Com SMOTE", color="#ff7f0e")

        plt.xticks(x, modelos, rotation=25, ha='right')
        plt.title(f"Comparativo de {titulo} - Antes vs Depois do SMOTE", fontsize=14)
        plt.ylabel(f"{titulo} (%)", fontsize=12)
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Adiciona rótulos de valor nas barras (em porcentagem)
        for bars in [bars1, bars2]:
            for bar in bars:
                altura = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, altura + 1,
                         f"{altura:.1f}%", ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

        # Adiciona variação Δ% acima das barras do SMOTE
        for i, delta in enumerate(delta_pct):
            cor = "#00ff7f" if delta > 0 else "#ff474c"
            sinal = "+" if delta > 0 else ""
            plt.text(x[i] + largura/2, smote_pct.iloc[i] + 4, f"{sinal}{delta:.1f}%",
                     ha='center', va='bottom', fontsize=9, color=cor, fontweight='bold')

        plt.tight_layout()
        nome_arquivo = f"plots/comparativos/Comparativo_{titulo.replace(' ', '_')}_SMOTE.png"
        plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"\nGráfico salvo em: {nome_arquivo}")

    # Plotar os comparativos desejados
    plot_comparativo("recall", "Recall")
    plot_comparativo("f1_score", "F1-Score")

    print("\nGráficos comparativos salvos em: plots/comparativos/")

def main():
    comparar_resultados_smote()

if __name__ == '__main__':
    main()