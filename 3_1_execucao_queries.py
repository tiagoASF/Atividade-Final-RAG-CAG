import os
import time
import asyncio
from pathlib import Path
import csv

os.environ["ENV_FASE"] = ".env3_1"

from dotenv import load_dotenv

from app.consulta import consultar
# from app.indexacao import _criar_lightrag
import app.config as config
from lightrag import QueryParam
from langfuse import observe
from langfuse import get_client
from lightrag.llm.openai import openai_complete_if_cache

load_dotenv(os.getenv("ENV_FASE", ".env3_1"))

from app.config import config_llm
api_key, modelo, base_url = config_llm()

if api_key is not None:
    os.environ["OPENAI_API_KEY"] = api_key

if base_url is not None:
    os.environ["OPENAI_BASE_URL"] = base_url

from app.log import obter_logger

log = obter_logger(__name__)

async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    api_key, modelo, base_url = config.config_llm()
    
    instrucao_pt = (
        "\n\nCRITICAL: Return all entity names, descriptions, relationship summaries, and metadata "
        "strictly in Brazilian Portuguese (pt-BR). Even if the prompt parameters or templates are in English, "
        "your textual output must be in Portuguese. Do not translate legal acronyms specific to the document "
        "context (e.g., PCDF, BO, IML), but all narrative must be strictly in Portuguese."
    )
    
    if system_prompt:
        system_prompt = system_prompt + instrucao_pt
    else:
        system_prompt = "You are a helpful assistant. " + instrucao_pt
    
    return await openai_complete_if_cache(
        modelo, 
        prompt, 
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=api_key, 
        base_url=base_url,
        temperature=0.0,
        **kwargs
    )

async def _criar_lightrag():
    from functools import partial
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed
    from lightrag.utils import EmbeddingFunc

    o_base, o_modelo = config.config_ollama()
    
    provider = config.provedor_llm()

    if provider == "ollama":
        # Configurações otimizadas para hardware local e limitações do Ollama (ex: nomic-embed)
        chunk_size = 600        # Menor fragmentação evita gargalo de GPU/CPU local e timeouts
        chunk_overlap = 100
        max_embed_tokens = 2048 # Limite rígido do nomic-embed-text para evitar erro de estouro
        log.info("Inicializando LightRAG com perfil LOCAL otimizado (Provedor: Ollama)")
    else:
        # Configurações robustas para nuvem (Grok / APIs comerciais)
        chunk_size = 1200       # Padrão original rico em contexto do LightRAG
        chunk_overlap = 150
        max_embed_tokens = 8192 # Janela ampla suportada nativamente na API do Grok
        log.info("Inicializando LightRAG com perfil de ALTO desempenho/janela (Provedor: Grok)")

     # Instanciação dinâmica do LightRAG
    rag = LightRAG(
        working_dir=str(config.PASTA_RAG_STORAGE), 
        llm_model_func=llm_func,
        chunk_token_size=chunk_size,
        chunk_overlap_token_size=chunk_overlap,
        embedding_func=EmbeddingFunc(
            embedding_dim=config.dimensao_embedding(),
            max_token_size=max_embed_tokens,
            func=partial(ollama_embed.func, embed_model=o_modelo, host=o_base)
        )
    )
    
    await rag.initialize_storages()
    return rag

# Configurando trace no langfuse
# @observe(name="3_execucao_queries_lightrag")
async def executar_consulta_lightrag(rag, pergunta):
    return await rag.aquery(pergunta, param=QueryParam(mode="hybrid"))

# @observe(name="3_execucao_queries_opensearch")
async def executar_consulta_opensearch(pergunta, tecnica):
    return await asyncio.to_thread(
        consultar, 
        pergunta, 
        "opensearch",
        5,          # top_k
        tecnica,    # técnica atual de consulta
        False       # rerank
    )

# Configurando trace no langfuse
async def main():
    t_inicio = time.time()
    arquivo_entrada = Path(__file__).resolve().parent / "datasets" / "perguntas_validacao.csv"
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    rag = await _criar_lightrag()
    # tecnicas_opensearch = ["baseline", "multi_query", "rag_fusion", "step_back"]
    tecnicas_opensearch = ["rag_fusion"]

    arquivos_saida = {}
    escritores = {}

    try:
        # 1. Abre os arquivos para todas as técnicas de consulta do OpenSearch com o chunking fixo campeão
        novo_cabecalho = ["Numero", "Pergunta", "Gabarito", "Resposta", "Tempo"]
        for tecnica in tecnicas_opensearch:
            nome_arquivo = f"3_opensearch_{tecnica}_chunking_fixo_validacao_perguntas.csv"
            caminho_saida = output_dir / nome_arquivo
            
            f_out = open(caminho_saida, mode='w', encoding='utf-8', newline='')
            escritor = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL, delimiter=',', quotechar='"')
            escritor.writerow(novo_cabecalho)
            
            arquivos_saida[tecnica] = f_out
            escritores[tecnica] = escritor
        
        # 2. Adiciona o arquivo do grafo (LightRAG)
        nome_arquivo_grafo = "3_lightrag_grafo_validacao_perguntas.csv"
        caminho_saida_grafo = output_dir / nome_arquivo_grafo
        
        f_out_grafo = open(caminho_saida_grafo, mode='w', encoding='utf-8', newline='')
        escritor_grafo = csv.writer(f_out_grafo, quoting=csv.QUOTE_MINIMAL, delimiter=',', quotechar='"')
        escritor_grafo.writerow(novo_cabecalho)
        
        arquivos_saida["grafo"] = f_out_grafo
        escritores["grafo"] = escritor_grafo

        print(f"Iniciando os testes lendo de '{arquivo_entrada}'...\n")
        with open(arquivo_entrada, mode='r', encoding='utf-8') as f_entrada:
            leitor = csv.reader(f_entrada, delimiter=',', quotechar='"')
            next(leitor, None)

            for linha in leitor:
                if len(linha) < 3:
                    continue
                    
                numero = linha[0]
                pergunta = linha[1]
                gabarito = linha[2]

                print(f"Processando Caso de Teste #{numero}: '{pergunta[:40]}...'")

                # Consulta do LightRAG (Grafo)
                tempo_inicio_grafo = time.perf_counter()
                try:
                    # resposta_lightRAG = await rag.aquery(pergunta, param=QueryParam(mode="hybrid"))
                    resposta_lightRAG = await executar_consulta_lightrag(rag, pergunta)
                except Exception as e:
                    log.error("Erro no LightRAG para o caso %s: %s", numero, e)
                    resposta_lightRAG = f"ERRO NA CONSULTA: {str(e)}"
                tempo_execucao_lightRAG = time.perf_counter() - tempo_inicio_grafo

                nova_linha_grafo = [
                    numero, pergunta, gabarito, 
                    resposta_lightRAG, f"{tempo_execucao_lightRAG:.4f}"
                ]
                escritores["grafo"].writerow(nova_linha_grafo)

                # Itera sobre todas as técnicas do OpenSearch usando a estratégia campeã de chunking fixo
                for tecnica in tecnicas_opensearch:
                    tempo_inicio_os = time.perf_counter()
                    try:
                        # resposta_opensearch, _, _ = await asyncio.to_thread(
                        #     consultar, 
                        #     pergunta, 
                        #     "opensearch",
                        #     5,          # top_k
                        #     tecnica,    # técnica atual de consulta
                        #     False       # rerank
                        # )
                        resposta_opensearch, _, _ = await executar_consulta_opensearch(pergunta, tecnica)
                    except Exception as e:
                        log.error("Erro ao processar caso %s com a técnica %s: %s", numero, tecnica, e)
                        resposta_opensearch = f"ERRO NA CONSULTA: {str(e)}"
                    
                    tempo_execucao_opensearch = time.perf_counter() - tempo_inicio_os

                    nova_linha_os = [
                        numero, pergunta, gabarito, 
                        resposta_opensearch, f"{tempo_execucao_opensearch:.4f}"
                    ]
                    escritores[tecnica].writerow(nova_linha_os)

                await asyncio.sleep(1.0)

        print(f"\nExecução finalizada com sucesso!")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{arquivo_entrada}' não foi encontrado. Verifique o caminho.")
    except Exception as e:
        print(f"Ocorreu um erro durante o processamento: {e}")
    finally:
        for f_out in arquivos_saida.values():
            f_out.close()
    
    t_total = time.time() - t_inicio
    print(f"Tempo total de execução: {t_total:.4f}s")

if __name__ == "__main__":
    asyncio.run(main())
    get_client().flush()
