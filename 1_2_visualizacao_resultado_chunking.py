import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Define o diretório de saída
pasta_output = Path("output")
pasta_output.mkdir(exist_ok=True)

# 1. Carregar o arquivo consolidado[cite: 5]
with open(pasta_output / "1_benchmark_ragas_chunking.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Converte o dicionário do JSON em DataFrame
df_raw = pd.DataFrame.from_dict(data, orient='index')

# 2. Tratamento para extrair o índice e o status do rerank
def parse_experimento(exp):
    if exp.endswith("_com_rerank"):
        return exp[:-len("_com_rerank")], "Com Rerank"
    elif exp.endswith("_sem_rerank"):
        return exp[:-len("_sem_rerank")], "Sem Rerank"
    return exp, "Desconhecido"

parsed_data = df_raw['experimento'].apply(parse_experimento)
df_raw['indice'] = [item[0] for item in parsed_data]
df_raw['rerank'] = [item[1] for item in parsed_data]

# Converte colunas numéricas
df_raw['context_precision'] = pd.to_numeric(df_raw['context_precision'], errors='coerce')
df_raw['context_recall'] = pd.to_numeric(df_raw['context_recall'], errors='coerce')

# 3. Calcula a média e transforma o formato para o Seaborn agrupar as 4 barras
df_agrupado = df_raw.groupby(['indice', 'rerank'])[['context_precision', 'context_recall']].mean().reset_index()

# Transforma de formato largo para longo, combinando Métrica e Rerank em uma única categoria de agrupamento (hue)
df_long = df_agrupado.melt(
    id_vars=['indice', 'rerank'], 
    value_vars=['context_precision', 'context_recall'],
    var_name='metrica', 
    value_name='score'
)

# Cria um rótulo amigável para a legenda seguindo a ordem desejada
def criar_legenda_ordenada(row):
    m = "Precision" if row['metrica'] == 'context_precision' else "Recall"
    return f"{m} ({row['rerank']})"

df_long['legenda'] = df_long.apply(criar_legenda_ordenada, axis=1)

# 4. Configurar o estilo visual e a paleta de cores personalizada
# Tons de azul para Precision (mais claro sem rerank, mais escuro com rerank)
# Tons de laranja para Recall (mais claro sem rerank, mais escuro com rerank)
palette_cores = {
    "Precision (Sem Rerank)": "#aec7e8",
    "Precision (Com Rerank)": "#1f77b4",
    "Recall (Sem Rerank)": "#ffbb78",
    "Recall (Com Rerank)": "#ff7f0e"
}

ordem_barras = [
    "Precision (Sem Rerank)", 
    "Precision (Com Rerank)", 
    "Recall (Sem Rerank)", 
    "Recall (Com Rerank)"
]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 7))

ax = sns.barplot(
    data=df_long, 
    x="indice", 
    y="score", 
    hue="legenda", 
    hue_order=ordem_barras,
    palette=palette_cores
)

plt.title("Comparativo Consolidado: Context Precision e Context Recall por Chunking", fontsize=16)
plt.xlabel("Estratégia de Chunking", fontsize=12)
plt.ylabel("Score Médio", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1.05)

# Posiciona a legenda fora do gráfico ou em um canto limpo para não sobrepor as barras
plt.legend(title="Métrica e Abordagem", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

# Salva o gráfico consolidado
caminho_fig = pasta_output / "1_comparativo_geral_precisao_recall.png"
plt.savefig(caminho_fig, dpi=300)
plt.close()
print(f"Gráfico consolidado gerado com sucesso: {caminho_fig}")

