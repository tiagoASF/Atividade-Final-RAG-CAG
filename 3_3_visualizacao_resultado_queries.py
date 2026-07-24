from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_grafico_comparativo():
    output_dir = Path(__file__).resolve().parent / "output"
    if not output_dir.exists():
        output_dir = Path(__file__).resolve().parent

    # Mapeamento dos arquivos com as respectivas técnicas/abordagens
    arquivos_metricas = {
        "OpenSearch Baseline": output_dir / "3_opensearch_baseline_chunking_fixo_validacao_perguntas.csv",
        "OpenSearch Multi-Query": output_dir / "3_opensearch_multi_query_chunking_fixo_validacao_perguntas.csv",
        "OpenSearch RAG Fusion": output_dir / "3_opensearch_rag_fusion_chunking_fixo_validacao_perguntas.csv",
        "OpenSearch Step-Back": output_dir / "3_opensearch_step_back_chunking_fixo_validacao_perguntas.csv",
        "LightRAG (Grafo)": output_dir / "3_lightrag_grafo_validacao_perguntas.csv"
    }

    dados_consolidados = []

    for tecnica, caminho in arquivos_metricas.items():
        if not caminho.exists():
            print(f"Aviso: Arquivo não encontrado para {tecnica} em {caminho}")
            continue
            
        df = pd.read_csv(caminho)
        
        # Calcula a média das métricas Ragas avaliadas
        media_precision = df["Context_Precision"].mean()
        media_recall = df["Context_Recall"].mean()
        media_tempo = pd.to_numeric(df["Tempo"], errors='coerce').mean()

        dados_consolidados.append({
            "Tecnica": tecnica,
            "Context Precision": media_precision,
            "Context Recall": media_recall,
            "Tempo Médio (s)": media_tempo
        })

    if not dados_consolidados:
        print("Nenhum dado encontrado para gerar o gráfico.")
        return

    df_resultado = pd.DataFrame(dados_consolidados)
    print("\nResumo das Métricas por Técnica:")
    print(df_resultado.to_string(index=False))

    # Configuração do Gráfico Comparativo (Barras Agrupadas)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Transforma o DataFrame para o formato longo (melt) para o seaborn
    df_melted = df_resultado.melt(
        id_vars=["Tecnica"], 
        value_vars=["Context Precision", "Context Recall"],
        var_name="Metrica", 
        value_name="Score"
    )

    ax = sns.barplot(
        data=df_melted, 
        x="Tecnica", 
        y="Score", 
        hue="Metrica", 
        palette="muted"
    )

    plt.title("Comparativo de Desempenho RAGAS por Técnica de Recuperação", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Técnica / Abordagem", fontsize=12, labelpad=10)
    plt.ylabel("Score Médio", fontsize=12, labelpad=10)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title="Métricas RAGAS", loc="upper right")

    # Adiciona os valores nas barras
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=9, color='black',
                xytext=(0, 3),
                textcoords='offset points'
            )

    plt.tight_layout()
    caminho_grafico = output_dir / "3_comparativo_tecnicas_ragas.png"
    plt.savefig(caminho_grafico, dpi=300)
    print(f"\nGráfico salvo com sucesso em: {caminho_grafico}")
    plt.show()

if __name__ == "__main__":
    gerar_grafico_comparativo()
