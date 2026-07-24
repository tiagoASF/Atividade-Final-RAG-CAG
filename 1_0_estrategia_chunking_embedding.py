import os
import json
import time
import asyncio
from pathlib import Path
import pandas as pd
from opensearchpy import OpenSearch
from datasets import Dataset
from dotenv import load_dotenv

# Configuração de ambiente
os.environ["ENV_FASE"] = ".env1"
load_dotenv(os.getenv("ENV_FASE"))

from app.main import _processar_ingestao
from app.config import config_opensearch

async def main():
    pasta_base = Path(__file__).resolve().parent
    arquivo = pasta_base / "datasets" / "pop_acolhimento_vitimas_violencia_domestica_limpo.pdf"
    pasta_output = pasta_base / "output"
    pasta_output.mkdir(parents=True, exist_ok=True)

    with open(arquivo, "rb") as f:
        conteudo_bytes = f.read()

    estrategias = ["fixo", "recursivo", "sentenca_janela", "semantico", "hierarquico"]
    indices_gerados = []

    for chunking in estrategias:
        t_inicio = time.time()
        nome_indice = os.getenv("OPENSEARCH_INDEX", "fase1_chunking") + f"_{chunking}"
        indices_gerados.append(nome_indice)
        
        relatorio = await asyncio.to_thread(_processar_ingestao, arquivo.name, conteudo_bytes, nome_indice, "opensearch", chunking)
        
        if relatorio is not None and not getattr(relatorio, 'erro', None):
            dados = relatorio.model_dump()
            dados["tempo_total"] = round(time.time() - t_inicio, 2)
            with open(pasta_output / f"1_{chunking}_embedding.json", "w", encoding="utf-8") as f_json:
                json.dump(dados, f_json, indent=4, ensure_ascii=False)
            print(f"Metadados de indexação salvos para {chunking}.")
        else:
            print(f"Erro ao processar técnica {chunking}: {getattr(relatorio, 'erro', 'Falha oculta')}")

if __name__ == "__main__":
    asyncio.run(main())

