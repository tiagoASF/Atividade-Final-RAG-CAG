"""
interface.py - Interface Gradio da bancada de avaliacao de embeddings.

A interface NAO calcula nada: ela conversa com a API (POST /avaliar) e apenas mostra os
resultados - tabela, graficos (barras por metrica + radar) e a explicacao do melhor modelo.

Rodar (2 terminais, de dentro de bench_embeddings/):
  1) uvicorn app.api:app --port 8000
  2) python interface.py        -> abre em http://localhost:7860
"""

import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from pathlib import Path

from dotenv import load_dotenv

PASTA_APP = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_APP.parent

load_dotenv(PASTA_PROJETO / os.getenv("ENV_FASE", ".env"))

API_URL = os.getenv("API_URL", "http://localhost:8080")

MODELOS_SUGERIDOS = ["nomic-embed-text", "bge-m3", "mxbai-embed-large", "snowflake-arctic-embed"]

def _metricas_cols(k):
    return [f"hit@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", "auc"]

def _metricas_labels(k):
    return [f"Hit@{k}", f"Recall@{k}", "MRR", f"NDCG@{k}", "AUC-ROC"]

def grafico_barras(validos, k):
    import numpy as np
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    n = len(validos)
    largura = 0.6 / max(1, n)
    x = np.arange(len(_metricas_cols(k)))
    
    cores_dos_modelos = []
    
    # 1. Plotagem das Métricas de Qualidade (Eixo Y Esquerdo)
    for i, r in enumerate(validos):
        # Busca no dicionário usando as chaves em minúsculo (_metricas_cols)
        valores = [r.get(c, 0) for c in _metricas_cols(k)]
        barras_metricas = ax1.bar(x + i * largura, valores, largura, label=f"{r['modelo']}")
        cores_dos_modelos.append(barras_metricas[0].get_facecolor())
    
    ax1.set_ylabel("Métricas de Qualidade (0 a 1)", color="black")
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x + largura * (n - 1) / 2)
    # Exibe os rótulos bonitos (_metricas_labels)
    ax1.set_xticklabels(_metricas_labels(k), rotation=15)
    
    # 2. Criando o Eixo Duplo para Latência (Eixo Y Direito)
    ax2 = ax1.twinx()
    
    posicao_latencia = len(_metricas_cols(k)) + 0.5
    largura_lat = 0.25
    
    modelos_nomes = [r["modelo"] for r in validos]
    latencias = [r.get("latencia_s", 0) for r in validos]
    
    x_lat = np.arange(len(modelos_nomes)) * largura_lat + posicao_latencia
    
    ax2.bar(
        x_lat, 
        latencias, 
        largura_lat, 
        color=cores_dos_modelos,
        alpha=0.7, 
        hatch="//", 
        edgecolor="white"
    )

    ax2.set_ylabel("Latência Total (segundos)", color="gray")
    ax1.set_xlim(-0.5, posicao_latencia + (n * largura_lat))
    
    # Ajustando as marcações do eixo X para incluir a seção de "Latência"
    ticks_x = list(x + largura * (n - 1) / 2) + [posicao_latencia + (largura_lat * (n - 1) / 2)]
    # Une as labels bonitas com a label da Latência
    labels_x = _metricas_labels(k) + ["Latência"]
    ax1.set_xticks(ticks_x)
    ax1.set_xticklabels(labels_x, rotation=15)
    
    ax1.set_title("Métricas de Qualidade vs. Latência por Modelo", pad=15)
    ax1.legend(loc="upper left", fontsize=8)
    
    fig.tight_layout()
    return fig


def grafico_radar(validos, k):
    import numpy as np
    
    cols_chaves = _metricas_cols(k)    # Para buscar do dicionário ("hit@10")
    cols_labels = _metricas_labels(k)  # Para plotar no gráfico ("Hit@10")
    
    ang = np.linspace(0, 2 * np.pi, len(cols_chaves), endpoint=False).tolist()
    ang += ang[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    
    for r in validos:
        # Busca no dicionário usando a chave real ("hit@10")
        vals = [r.get(c, 0) for c in cols_chaves]
        vals += vals[:1]
        ax.plot(ang, vals, label=r["modelo"])
        ax.fill(ang, vals, alpha=0.1)
        
    ax.set_xticks(ang[:-1])
    # Define as labels de visualização corretas ("Hit@10")
    ax.set_xticklabels(cols_labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Comparação (radar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def rodar(modelos_sel, modelos_extra, k, fonte, arquivo):
    modelos = list(modelos_sel or [])
    if modelos_extra:
        modelos += [m.strip() for m in modelos_extra.split(",") if m.strip()]
    if not modelos:
        return pd.DataFrame(), None, None, "Selecione ao menos um modelo."

    dataset_json = None
    if fonte == "upload" and arquivo:
        with open(arquivo, "r", encoding="utf-8") as f:
            dataset_json = f.read()

    try:
        r = requests.post(f"{API_URL}/avaliar",
                          json={"modelos": modelos, "k": int(k), "dataset_json": dataset_json},
                          timeout=1800)
        r.raise_for_status()
        res = r.json()
    except Exception as e:
        return pd.DataFrame(), None, None, f"Erro ao chamar a API ({API_URL}): {e}"

    validos = [x for x in res["resultados"] if "erro" not in x]
    erros = [f"- {x['modelo']}: {x['erro']}" for x in res["resultados"] if "erro" in x]
    if not validos:
        return pd.DataFrame(), None, None, "Nenhum modelo avaliado.\n" + "\n".join(erros)

    cols = ["modelo", "dim", "latencia_s"] + _metricas_cols(k)
    df = pd.DataFrame([{c: v.get(c) for c in cols} for v in validos])
    md = f"### Melhor modelo: **{res.get('melhor')}**\n\n{res.get('explicacao','')}"
    if erros:
        md += "\n\n**Falhas:**\n" + "\n".join(erros)
    return df, grafico_barras(validos, k), grafico_radar(validos, k), md

with gr.Blocks(title="Bancada de Embeddings") as demo:
    gr.Markdown("# Qual o melhor modelo de embedding para o seu corpus?\n"
                "Compara modelos do Ollama por **retrieval (Hit@K, Recall@K)**, "
                "**ranking (MRR, NDCG@K)** e **separabilidade (AUC-ROC)**.")
    with gr.Row():
        with gr.Column():
            modelos_sel = gr.CheckboxGroup(MODELOS_SUGERIDOS, value=["nomic-embed-text", "bge-m3"],
                                           label="Modelos (Ollama) a comparar")
            modelos_extra = gr.Textbox(label="Outros modelos (separados por virgula)", placeholder="ex.: all-minilm")
            k = gr.Slider(1, 20, value=10, step=1, label="K (top-K)")
            fonte = gr.Radio(["padrao", "upload"], value="padrao", label="Dataset")
            arquivo = gr.File(label="JSON do aluno (corpus + queries com gabarito)", type="filepath")
            botao = gr.Button("Avaliar", variant="primary")
        with gr.Column():
            explicacao = gr.Markdown()
    tabela = gr.Dataframe(label="Metricas por modelo")
    with gr.Row():
        g_barras = gr.Plot(label="Barras por metrica")
        g_radar = gr.Plot(label="Radar")
    botao.click(rodar, [modelos_sel, modelos_extra, k, fonte, arquivo],
                [tabela, g_barras, g_radar, explicacao])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT", "7860")))
