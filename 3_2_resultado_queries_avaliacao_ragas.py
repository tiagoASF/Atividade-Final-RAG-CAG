import os
import time
import asyncio
from pathlib import Path
import csv
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

os.environ["ENV_FASE"] = ".env3_2"
load_dotenv(os.getenv("ENV_FASE"))

from ragas import evaluate, RunConfig
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import ContextPrecision, ContextRecall
from typing import cast
from langfuse import observe
from langfuse import get_client

# Configurando trace no langfuse
@observe(name="3_avaliacao_ragas_queries")
async def avaliar_arquivos_saida():
    output_dir = Path(__file__).resolve().parent / "output"
    
    # Lista todos os arquivos CSV gerados na etapa anterior
    arquivos_csv = list(output_dir.glob("3_*_validacao_perguntas.csv"))
    
    if not arquivos_csv:
        print("Nenhum arquivo de validação encontrado na pasta output.")
        return

    # 1. Configuração do Cliente Juiz (OpenRouter / LLM configurado)
    chat_model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
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

    for arquivo_path in arquivos_csv:
        print(f"\nProcessando e avaliando: {arquivo_path.name}")
        df = pd.read_csv(arquivo_path)
        
        user_inputs = []
        retrieved_contexts = []
        references = []
        
        for _, linha in df.iterrows():
            user_inputs.append(str(linha["Pergunta"]))
            # O Ragas exige 'retrieved_contexts' como uma lista de textos por consulta
            retrieved_contexts.append([str(linha["Resposta"])])
            references.append(str(linha["Gabarito"]))
            
        # Alinhado exatamente à exigência do Ragas para ContextPrecision e ContextRecall
        dataset_hf = Dataset.from_dict({
            "user_input": user_inputs,
            "retrieved_contexts": retrieved_contexts,
            "reference": references
        })
        
        t_inicio_ragas = time.perf_counter()
        try:
            resultado = evaluate(
                dataset=dataset_hf,
                metrics=metricas,
                llm=llm_ragas,
                run_config=config
            )
            scores = resultado.scores
            
            context_precisions = []
            context_recalls = []
            
            for item in scores:
                context_precisions.append(item.get("context_precision", 0.0))
                context_recalls.append(item.get("context_recall", 0.0))
                
            tempo_ragas_total = time.perf_counter() - t_inicio_ragas
            tempo_por_linha = tempo_ragas_total / len(df) if len(df) > 0 else 0.0
            
            df["Context_Precision"] = context_precisions
            df["Context_Recall"] = context_recalls
            df["Tempo_Ragas"] = [f"{tempo_por_linha:.4f}"] * len(df)
            print(f"Sucesso na avaliação Ragas para {arquivo_path.name}")
            
        except Exception as e:
            print(f"Erro ao rodar Ragas no arquivo {arquivo_path.name}: {e}")
            df["Context_Precision"] = 0.0
            df["Context_Recall"] = 0.0
            df["Tempo_Ragas"] = "0.0000"

        # Salva o arquivo atualizado substituindo o anterior na pasta output
        df.to_csv(arquivo_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        print(f"Arquivo atualizado com as métricas Ragas: {arquivo_path.name}")

    print("\nToda a avaliação via Ragas foi concluída com sucesso para os arquivos de validação!")

if __name__ == "__main__":
    asyncio.run(avaliar_arquivos_saida())
    get_client().flush()
