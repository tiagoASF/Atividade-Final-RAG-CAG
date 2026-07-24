import os
import json
import asyncio
from pathlib import Path
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Configuração de ambiente para a Fase 1
os.environ["ENV_FASE"] = ".env1"
load_dotenv(os.getenv("ENV_FASE"))

# Importa a função de construção de busca avançada do projeto
from app import busca_avancada

# Componentes de avaliação do Ragas (Alinhado à v0.4.3)
from ragas import evaluate, RunConfig
from ragas.llms import BaseRagasLLM
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import ContextPrecision, ContextRecall
from langfuse.langchain import CallbackHandler

from typing import cast

def executar_busca_pipeline(query_text, indice, rerank_status):
    """Executa a busca utilizando o pipeline Haystack oficial definido em busca_avancada.py"""
    try:
        # Define temporariamente o índice do OpenSearch via variável de ambiente que o busca_avancada lê
        os.environ["OPENSEARCH_INDEX"] = indice
        
        # Constrói o pipeline utilizando a técnica 'baseline' (busca densa pura) e o estado do rerank
        pipe, inputs, chave_docs = busca_avancada.construir(
            tecnica="baseline", 
            top_k=5, 
            pergunta=query_text, 
            rerank=rerank_status
        )
        
        # Adiciona a query ao dicionário de inputs se o rerank estiver habilitado
        if rerank_status:
            inputs["reranker"] = {"query": query_text}
        
        # Executa o pipeline Haystack
        saida = pipe.run(inputs, include_outputs_from={chave_docs})
        docs = saida[chave_docs]["documents"]
        
        # Extrai o conteúdo dos documentos retornados
        return [d.content for d in docs]
    except Exception as e:
        print(f"Erro ao executar pipeline para o índice {indice} (rerank={rerank_status}): {e}")
        return []

async def rodar_avaliacao_ragas(pasta_output, lista_indices):
    print("\n" + "="*50)
    print("INICIANDO AVALIAÇÃO VIA RAGAS (Pipeline Haystack: Com e Sem Rerank)")
    print("="*50)

    # 1. Configuração do Cliente Juiz (OpenRouter)
    chat_model = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        max_tokens=3000
    )

    llm_ragas = LangchainLLMWrapper(chat_model)
    llm_for_metrics = cast(BaseRagasLLM, llm_ragas)
    
    precision = ContextPrecision(llm=llm_for_metrics)
    recall = ContextRecall(llm=llm_for_metrics)
    metricas = [precision, recall]

    config = RunConfig(max_workers=2)
    
    path_benchmark = Path(__file__).resolve().parent / "datasets" / "corpus_juridico_validacao_chunking.json"
    with open(path_benchmark, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
        
    documentos_map = {doc["id"]: doc["texto"] for doc in benchmark_data["documentos"]}
    queries = benchmark_data["queries_benchmark"]
    resultados_finais = {}

    langfuse_handler = CallbackHandler()
    
    # Define as variações de avaliação para cada índice (Sem Rerank e Com Rerank)
    modos_avaliacao = [
        {"nome_sufijo": "sem_rerank", "usar_rerank": False},
        {"nome_sufijo": "com_rerank", "usar_rerank": True}
    ]

    for indice_base in lista_indices:
        for modo in modos_avaliacao:
            nome_experimento = f"{indice_base}_{modo['nome_sufijo']}"
            usar_rerank = modo["usar_rerank"]
            
            print(f"\nAvaliando experimento: {nome_experimento}...")
            dados_ragas = {"user_input": [], "retrieved_contexts": [], "reference": []}
            
            for q in queries:
                query_text = q["query"]
                doc_ids_esperados = list(q["relevancia"].keys())
                textos_referencia = [documentos_map[d_id] for d_id in doc_ids_esperados if d_id in documentos_map]
                
                if not textos_referencia: 
                    continue
                
                # Executa a busca através do pipeline centralizado de busca_avancada
                chunks_recuperados = executar_busca_pipeline(query_text, indice_base, usar_rerank)
                
                dados_ragas["user_input"].append(query_text)
                dados_ragas["retrieved_contexts"].append(chunks_recuperados)
                dados_ragas["reference"].append("\n".join(textos_referencia))
                
            dataset_hf = Dataset.from_dict(dados_ragas)
            
            resultado = evaluate(
                dataset=dataset_hf,
                metrics=metricas,
                llm=llm_ragas,
                run_config=config,
                callbacks=[langfuse_handler]
            )

            resultados_finais[nome_experimento] = resultado
        
    dados_para_df = []
    for nome_experimento, resultado in resultados_finais.items():
        lista_scores = resultado.scores
        for item in lista_scores:
            item_com_experimento = item.copy()
            item_com_experimento['experimento'] = nome_experimento
            dados_para_df.append(item_com_experimento)

    df_comparativo = pd.DataFrame(dados_para_df)
    df_comparativo.to_json(pasta_output / "1_benchmark_ragas_chunking.json", indent=4, orient="index")
    print(f"\nAvaliação salva com sucesso em '{pasta_output / '1_benchmark_ragas_chunking.json'}'!")

async def main():
    pasta_base = Path(__file__).resolve().parent
    pasta_output = pasta_base / "output"
    pasta_output.mkdir(parents=True, exist_ok=True)

    indices_gerados = [
        "fase1_chunking_fixo", 
        "fase1_chunking_recursivo", 
        "fase1_chunking_sentenca_janela",
        "fase1_chunking_semantico", 
        "fase1_chunking_hierarquico"
    ]

    await rodar_avaliacao_ragas(pasta_output, indices_gerados)

if __name__ == "__main__":
    asyncio.run(main())

