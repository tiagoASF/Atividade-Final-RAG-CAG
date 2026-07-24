import os
import json
import time
import asyncio
from pathlib import Path
import matplotlib.pyplot as plt

os.environ["ENV_FASE"] = ".env0"

from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_FASE", ".env0"))

from bench_embeddings.app.dados import carregar_json
from bench_embeddings.app.avaliacao import avaliar
from bench_embeddings.interface import grafico_barras, grafico_radar

# Lista de modelos que você já validou no seu Ollama
MODELOS = [
    "nomic-embed-text", 
    "bge-m3", 
    "bge-large:latest", 
    "mxbai-embed-large",
    "snowflake-arctic-embed:latest", 
    "granite-embedding:278m",
    "paraphrase-multilingual:latest"
]

def avaliar_e_salvar_por_k(k, cids, ctexts, queries, pasta_output):
    """
    Executa a avaliação para um valor de k específico, gera o JSON e os gráficos correspondentes.
    """
    t_inicio_k = time.time()
    print(f"\n=========================================")
    print(f"🚀 Iniciando avaliação para K = {k}...")
    print(f"=========================================")

    # Define os caminhos de saída dinâmicos baseados no valor de k
    arquivo_resultado = pasta_output / f"0_k{k}_embedding.json"
    caminho_barras = pasta_output / f"0_k{k}_grafico_barras.png"
    caminho_radar = pasta_output / f"0_k{k}_grafico_radar.png"

    # 1. Executa a avaliação
    resultado = avaliar(cids, ctexts, queries, MODELOS, k)
    resultado["corpus"] = len(cids)
    resultado["queries"] = len(queries)

    # 2. Salva o arquivo JSON
    with open(arquivo_resultado, "w", encoding="utf-8") as f_json:
        json.dump(resultado, f_json, indent=4, ensure_ascii=False)
    print(f"✅ Resultado JSON salvo em: {arquivo_resultado}")

    # 3. Geração de Gráficos
    validos = [r for r in resultado["resultados"] if "erro" not in r]

    if validos:
        print(f"📊 Gerando gráficos para K = {k}...")
        # Usa as funções importadas do módulo interface
        fig_barras = grafico_barras(validos, k)
        fig_radar = grafico_radar(validos, k)

        # Salva as figuras diretamente no diretório output
        fig_barras.savefig(caminho_barras, dpi=150, bbox_inches="tight")
        fig_radar.savefig(caminho_radar, dpi=150, bbox_inches="tight")

        # Fecha as figuras para evitar acúmulo de memória do Matplotlib
        plt.close(fig_barras)
        plt.close(fig_radar)

        print(f"🖼️  Gráfico de barras salvo em: {caminho_barras}")
        print(f"🖼️  Gráfico radar salvo em: {caminho_radar}")
    else:
        print("⚠️  Nenhum resultado válido para plotar.")

    print(f"⏱️ Tempo para K={k}: {time.time() - t_inicio_k:.2f}s")


async def main():
    t_inicio = time.time()

    diretorio_atual = Path(__file__).resolve().parent
    arquivo_dataset = diretorio_atual / "datasets" / "corpus_juridico_benchmark.json"
    pasta_output = diretorio_atual / "output"
    
    # Garante que a pasta output exista fisicamente
    pasta_output.mkdir(parents=True, exist_ok=True)

    print("Carregando o dataset de benchmark...")
    with open(arquivo_dataset, "r", encoding="utf-8") as f:
        dataset_json = f.read()
    
    cids, ctexts, queries = carregar_json(dataset_json)
    
    # Defina aqui os valores de K que você deseja testar em lote
    valores_k_para_testar = [3, 5, 10]
    
    for k in valores_k_para_testar:
        avaliar_e_salvar_por_k(k, cids, ctexts, queries, pasta_output)

    print("\n=========================================")
    print(f"🎉 Todo o processo foi finalizado em {time.time() - t_inicio:.2f}s!")
    print(f"Os resultados estão disponíveis na pasta: {pasta_output}")
    print("=========================================")

if __name__ == "__main__":
    asyncio.run(main())