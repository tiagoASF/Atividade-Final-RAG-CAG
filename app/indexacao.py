"""
indexacao.py - Decide ESTRATEGIA de indexacao (destino + chunking) e grava no storage.

Decisoes TRANSPARENTES (heuristicas, sem LLM):
  DESTINO  : 'grafo' (LightRAG) se texto longo e rico em entidades (multi-hop);
             senao 'opensearch'. Override: 'auto' | 'opensearch' | 'grafo'.

  CHUNKING (so quando destino=opensearch) - escolhe a melhor das tecnicas:
     - tabela          : documento tem tabelas -> 1 chunk por tabela (preserva estrutura)
     - hierarquico     : documento estruturado em secoes (varios titulos)
     - sentenca_janela : texto de lei / denso em artigos -> precisao por sentenca
     - semantico       : texto longo e heterogeneo -> corta em mudancas de topico
     - recursivo       : texto corrido (default robusto: respeita paragrafos/sentencas)
     - fixo            : documento curto / baseline
   Override: 'auto' | fixo | recursivo | sentenca_janela | semantico | hierarquico.

Cada tecnica usa um componente NATIVO do Haystack. Embedding: Ollama (nomic-embed-text).
"""

import asyncio
import math
import re
from functools import partial

from haystack import Document

from . import config
from .log import obter_logger

log = obter_logger(__name__)

LIMIAR_ENTIDADES = 30
MIN_PALAVRAS_GRAFO = 800
TECNICAS_CHUNK = {"fixo", "recursivo", "sentenca_janela", "semantico", "hierarquico"}

import os
from dotenv import load_dotenv
from pathlib import Path

PASTA_APP = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_APP.parent

load_dotenv(PASTA_PROJETO / os.getenv("ENV_FASE", ".env"))


# ---------------------------------------------------------------------------
# DESTINO (OpenSearch vs LightRAG)
# ---------------------------------------------------------------------------
def _entidades_distintas(conteudo):
    palavras = conteudo.split()
    if len(palavras) < MIN_PALAVRAS_GRAFO:
        return 0
    caps = {w.strip(".,;:()") for w in palavras if w[:1].isupper() and len(w) > 2}
    return len(caps)


def decidir_destino(dados, override="auto"):
    if override in ("opensearch", "grafo"):
        log.info("Destino forcado pelo usuario: %s", override)
        return override, f"forcado pelo usuario (override={override})"
    n_ent = _entidades_distintas(dados.get("conteudo", ""))
    log.debug("Heuristica de destino: %d entidades distintas (limiar=%d)", n_ent, LIMIAR_ENTIDADES)
    if n_ent >= LIMIAR_ENTIDADES:
        return "grafo", f"texto longo e rico em entidades ({n_ent} distintas >= {LIMIAR_ENTIDADES}) -> multi-hop"
    return "opensearch", f"texto/tabela direto ({n_ent} entidades distintas < {LIMIAR_ENTIDADES})"


# ---------------------------------------------------------------------------
# AVALIADOR de chunking (so para OpenSearch)
# ---------------------------------------------------------------------------
def _n_titulos(c):
    return len([l for l in c.splitlines() if l.lstrip().startswith("#")])


def _n_artigos(c):
    return len(re.findall(r"\bArt\.?\s*\d+", c))


def avaliar_chunking(dados, override="auto"):
    """Escolhe a melhor tecnica de chunking pela ESTRUTURA do documento (explicavel)."""
    conteudo = dados.get("conteudo", "")
    if dados.get("tabelas"):
        return "tabela", "documento tem tabelas -> 1 chunk por tabela (preserva a estrutura)"
    if override in TECNICAS_CHUNK:
        return override, f"forcado pelo usuario (chunking={override})"

    n_pal, n_tit, n_art = len(conteudo.split()), _n_titulos(conteudo), _n_artigos(conteudo)
    log.debug("Sinais de chunking: %d palavras, %d titulos, %d 'Art.'", n_pal, n_tit, n_art)
    if n_tit >= 3:
        return "hierarquico", f"documento estruturado em secoes ({n_tit} titulos) -> hierarquico"
    if n_art >= 5:
        return "sentenca_janela", f"texto de lei/denso em artigos ({n_art} 'Art.') -> precisao por sentenca"
    if n_pal >= 1500:
        return "semantico", f"texto longo e heterogeneo ({n_pal} palavras) -> cortes por mudanca de topico"
    if n_pal >= 300:
        return "recursivo", f"texto corrido ({n_pal} palavras) -> recursivo (respeita paragrafos/sentencas)"
    return "fixo", f"documento curto ({n_pal} palavras) -> fixo (baseline)"


# ---------------------------------------------------------------------------
# Chunkers (componentes NATIVOS do Haystack)
# ---------------------------------------------------------------------------
def _rodar(splitter, conteudo):
    if hasattr(splitter, "warm_up"):
        splitter.warm_up()
    return splitter.run(documents=[Document(content=conteudo)])["documents"]


def _ollama_doc_embedder():
    from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
    base_url, modelo = config.config_ollama()
    return OllamaDocumentEmbedder(model=modelo, url=base_url)


def chunkar(conteudo, tecnica):
    from haystack.components.preprocessors import (DocumentSplitter,
        EmbeddingBasedDocumentSplitter, HierarchicalDocumentSplitter,
        RecursiveDocumentSplitter)

    log.debug("Chunkando com tecnica '%s' (%d caracteres)", tecnica, len(conteudo))
    if tecnica == "fixo":
        return _rodar(DocumentSplitter(split_by="word", split_length=200, split_overlap=0), conteudo)
    if tecnica == "recursivo":
        return _rodar(RecursiveDocumentSplitter(split_length=200, split_overlap=30, split_unit="word"), conteudo)
    if tecnica == "sentenca_janela":
        # janela = grupos de sentencas com sobreposicao (sentence-window)
        return _rodar(DocumentSplitter(split_by="sentence", split_length=3, split_overlap=1), conteudo)
    if tecnica == "semantico":
        sp = EmbeddingBasedDocumentSplitter(document_embedder=_ollama_doc_embedder(),
                                            sentences_per_group=3, language="pt")
        return _rodar(sp, conteudo)
    if tecnica == "hierarquico":
        nos = _rodar(HierarchicalDocumentSplitter(block_sizes={400, 100}, split_by="word"), conteudo)
        # indexa apenas as FOLHAS (chunks pequenos) - estrutura-aware
        return [d for d in nos if not d.meta.get("__children_ids")] or nos
    # 'tabela' ou desconhecida -> doc inteiro (tabela em markdown ja vem estruturada)
    return [Document(content=conteudo)]


# ---------------------------------------------------------------------------
# Gravacao: OpenSearch
# ---------------------------------------------------------------------------
def _store_opensearch(nome_indice):
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
    os_cfg = config.config_opensearch()
    auth = (os_cfg["usuario"], os_cfg["senha"]) if os_cfg["usuario"] else None
    return OpenSearchDocumentStore(hosts=os_cfg["url"], index=nome_indice,
                                   embedding_dim=config.dimensao_embedding(),
                                   http_auth=auth, use_ssl=False, verify_certs=False)


def indexar_opensearch(docs, meta, nome_indice):
    for d in docs:
        d.meta.update(meta)
    log.info("Gerando embeddings (Ollama) para %d chunk(s)...", len(docs))
    embedder = _ollama_doc_embedder()
    if hasattr(embedder, "warm_up"):
        embedder.warm_up()
    docs_emb = embedder.run(documents=docs)["documents"]
    log.info("Gravando %d documento(s) no OpenSearch (indice '%s')...",
             len(docs_emb), config.config_opensearch()["indice"])
    _store_opensearch(nome_indice).write_documents(docs_emb)
    return len(docs_emb)


# ---------------------------------------------------------------------------
# Gravacao: LightRAG (grafo)
# ---------------------------------------------------------------------------
# async def _criar_lightrag():
#     from lightrag import LightRAG
#     from lightrag.llm.ollama import ollama_embed
#     from lightrag.llm.openai import openai_complete_if_cache
#     from lightrag.utils import EmbeddingFunc

#     api_key, modelo, base_url = config.config_llm()
#     o_base, o_modelo = config.config_ollama()

#     async def _llm_model_func_custom(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
#         """Funcao customizada que intercepta as chamadas do LightRAG para o Ollama e injeta as regras em PT-BR."""
#         from lightrag.llm.openai import openai_complete_if_cache

#         # --- REGRA INJETADA: Força o comportamento em português direto no System Prompt ---
#         instrucao_pt = (
#             "\n\nCRITICAL: Return all entity names, descriptions, relationship summaries, and metadata "
#             "strictly in Brazilian Portuguese (pt-BR). Even if the prompt parameters or templates are in English, "
#             "your textual output must be in Portuguese. Do not translate legal acronyms specific to the document "
#             "context (e.g., PCDF, BO, IML), but all narrative must be strictly in Portuguese."
#         )
        
#         if system_prompt:
#             system_prompt = system_prompt + instrucao_pt
#         else:
#             system_prompt = "You are a helpful assistant. " + instrucao_pt

#         # Aponta para o seu Ollama local usando o mesmo helper compatível com OpenAI do LightRAG
#         # (Ajuste o nome do modelo 'gemma2' se estiver usando outro no config)
#         return await openai_complete_if_cache(
#             model=getattr(config, "LLM_MODEL", "gemma2"),
#             prompt=prompt,
#             system_prompt=system_prompt,
#             history_messages=history_messages or [],
#             api_key="ollama",                     # O Ollama não exige chave real, mas a lib pede preenchimento
#             base_url="http://localhost:11434/v1", # Endpoint padrão da API OpenAI-compatible do Ollama
#             **kwargs
#         )

#     async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
#         return await openai_complete_if_cache(modelo, prompt, system_prompt=system_prompt,
#                                               history_messages=history_messages or [],
#                                               api_key=api_key, base_url=base_url,
#                                               temperature=0.0,
#                                               **kwargs)

#     rag = LightRAG(working_dir=str(config.PASTA_RAG_STORAGE), llm_model_func=llm_func,
#                    embedding_func=EmbeddingFunc(embedding_dim=config.dimensao_embedding(),
#                        max_token_size=8192,
#                        func=partial(ollama_embed.func, embed_model=o_modelo, host=o_base)))
#     await rag.initialize_storages()
#     return rag

# ---------------------------------------------------------------------------
# Gravacao: LightRAG (grafo)
# ---------------------------------------------------------------------------
async def _criar_lightrag():
    from functools import partial
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    api_key, modelo, base_url = config.config_llm()
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

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        # --- INJETADO: Força o comportamento em português direto no System Prompt ---
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


def rodar_async(coro_factory):
    """Roda uma corrotina com seguranca, HAJA ou NAO um event loop ativo.

    O LightRAG e assincrono. Chamar asyncio.run() dentro de um endpoint 'async def'
    do FastAPI quebra ('asyncio.run() cannot be called from a running event loop').
    Aqui: se nao ha loop, usa asyncio.run direto; se ja ha um loop rodando, executa
    numa thread separada (que tem o proprio loop).
    """
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())  # sem loop ativo (caso comum)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro_factory())).result()


def indexar_grafo(conteudo):
    resultado = {"n_chunks": 0}
    async def _run():
        log.info("Construindo grafo no LightRAG (varias chamadas de LLM, pode demorar)...")
        rag = await _criar_lightrag()

        # 1) Tenta acessar o armazenamento de chunks brutos de texto do LightRAG
        # Ele costuma ficar em rag.chunks_vdb ou rag.full_chunks
        storage_chunks = None
        for attr in ['chunks_vdb', 'full_chunks', 'text_chunks']:
            if hasattr(rag, attr):
                storage_chunks = getattr(rag, attr)
                break
        
        # Conta quantos chunks existiam antes
        chunks_antes = 0
        if storage_chunks:
            # Se for um Vector Storage (como chunks_vdb) ou KVStorage
            if hasattr(storage_chunks, 'get_all_keys'):
                chunks_antes = len(await storage_chunks.get_all_keys())
            elif hasattr(storage_chunks, '_client') and hasattr(storage_chunks._client, 'get_all_keys'):
                chunks_antes = len(await storage_chunks._client.get_all_keys())

        try:
            await rag.ainsert(conteudo)
        finally:
            await rag.finalize_storages()
        
        # 2) Conta quantos chunks existem depois
        chunks_depois = 0
        if storage_chunks:
            if hasattr(storage_chunks, 'get_all_keys'):
                chunks_depois = len(await storage_chunks.get_all_keys())
            elif hasattr(storage_chunks, '_client') and hasattr(storage_chunks._client, 'get_all_keys'):
                chunks_depois = len(await storage_chunks._client.get_all_keys())
    
        # Calcula a diferença
        diferenca = chunks_depois - chunks_antes
        
        # 3) SE AINDA ASSIM DER ZERO: Fallback inspecionando o arquivo físico de log de chunks
        if diferenca <= 0:
            import json
            import os
            # O LightRAG salva o histórico completo em um JSON no diretório de storage
            caminho_kv = os.path.join(config.PASTA_RAG_STORAGE, "kv_store_full_chunks.json")
            if os.path.exists(caminho_kv):
                try:
                    with open(caminho_kv, 'r', encoding='utf-8') as f:
                        dados_kv = json.load(f)
                        # Como o LightRAG coloca o hash do doc nas chaves, estimamos 
                        # ou pegamos o total (aqui pegamos a diferença se tivéssemos o histórico)
                        # Para garantir, se o log diz 10, vamos tentar buscar chaves recentes ou estimar pelo conteúdo:
                        pass
                except Exception:
                    pass

        # Se a diferença capturada via código foi válida, usamos ela
        resultado["n_chunks"] = diferenca if diferenca > 0 else 0

    rodar_async(_run)
    log.info("Grafo atualizado no LightRAG (storage: %s)", config.PASTA_RAG_STORAGE)
    
    # Se a inspeção dinâmica falhou (retornou 0), vamos usar uma heurística baseada no 
    # tamanho do texto para estimar os chunks criados (o LightRAG usa padrão de token_string_spliter)
    if resultado["n_chunks"] == 0:
        # Padrão do LightRAG: chunk_size=1200 tokens (aprox 4000 a 4500 caracteres)
        tamanho_texto = len(conteudo)
        estimativa = max(1, math.ceil(tamanho_texto / 4000))
        log.warning("Não foi possível ler as chaves do Storage do LightRAG. Estimando %d chunks pelo tamanho do texto.", estimativa)
        return estimativa
    
    return resultado["n_chunks"] if resultado["n_chunks"] > 0 else 1

# ---------------------------------------------------------------------------
# Orquestra a indexacao
# ---------------------------------------------------------------------------
def indexar(dados, nome_indice, meta, destino_override="auto", chunking_override="auto"):
    destino, motivo_destino = decidir_destino(dados, destino_override)
    log.info("Destino de indexacao: %s (%s)", destino, motivo_destino)
    if destino == "grafo":
        n = indexar_grafo(dados.get("conteudo", ""))
        return {"destino": destino, "motivo_destino": motivo_destino,
                "chunking": "grafo (LightRAG gerencia)", "motivo_chunking": "destino=grafo",
                "n_chunks": n}
    tecnica, motivo_chunking = avaliar_chunking(dados, chunking_override)
    log.info("Tecnica de chunking: %s (%s)", tecnica, motivo_chunking)
    docs = chunkar(dados.get("conteudo", ""), tecnica)
    n = indexar_opensearch(docs, meta, nome_indice)
    log.info("Indexacao concluida: %d chunk(s) no OpenSearch", n)
    return {"destino": destino, "motivo_destino": motivo_destino,
            "chunking": tecnica, "motivo_chunking": motivo_chunking, "n_chunks": n}
