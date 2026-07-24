import os
import json
import time
import asyncio
from pathlib import Path

os.environ["ENV_FASE"] = ".env2"

from app.main import _processar_ingestao

# CHUNKING_OPCOES = ["auto", "fixo", "recursivo", "sentenca_janela", "semantico", "hierarquico"]
# DESTINO_OPCOES = ["auto", "opensearch", "grafo"]
# TECNICA_OPCOES = ["baseline", "multi_query", "rag_fusion", "step_back"]


"""Recebe um documento, decide como extrair e indexar, e devolve o relatorio.

estrategia: auto | opensearch | grafo   (destino da indexacao)
chunking:   auto | fixo | recursivo | sentenca_janela | semantico | hierarquico
"""

async def main():
    t_inicio = time.time()
    arquivo = Path(__file__).resolve().parent / "datasets" / "pop_acolhimento_vitimas_violencia_domestica_limpo.pdf"

    with open(arquivo, "rb") as f:
        conteudo_bytes = f.read()

    # Observação: o índice fase2_grafo não será criado, já que o LightRAG não utiliza o OpenSearch.
    # Ele está aqui apenas para compatibilidade com a interface de ingestão, que foi alterada
    # para gerar os chunkings em lote no script 1_0_estrategia_chunking_embedding.py.
    relatorio = await asyncio.to_thread(
        _processar_ingestao, 
        arquivo.name, 
        conteudo_bytes, 
        os.getenv("OPENSEARCH_INDEX", "fase2_grafo"),
        "grafo", 
        "auto"
    )
    
    t_total = time.time() - t_inicio

    print("Processamento concluído!")

    if relatorio.ok:
        dados_salvamento = relatorio.model_dump()

        dados_salvamento["tempo_total"] = round(t_total, 2)

        print(dados_salvamento)

        with open(Path(__file__).resolve().parent / "output" / "2_grafo_LightRAG_embedding.json", "w", encoding="utf-8") as f_json:
            # certas_ascii=False garante que acentos e caracteres em português fiquem legíveis no arquivo
            json.dump(dados_salvamento, f_json, indent=4, ensure_ascii=False)
    else:
        print("Erro ao processar o documento.")

if __name__ == "__main__":
    asyncio.run(main())

