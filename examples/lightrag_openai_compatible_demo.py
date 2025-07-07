import os
import asyncio
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./dickens"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_compatible_demo.log"))

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
        **kwargs,
    )


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


async def insert_documents(rag, file_paths):
    """
    Insère un ou plusieurs documents dans la base de connaissances RAG.
    
    Args:
        rag: Instance de LightRAG
        file_paths: Dictionnaire des fichiers à insérer avec leur doc_id
                  Exemple: {"livre1": "chemin/vers/livre1.txt", ...}
    """
    for doc_id, file_path in file_paths.items():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                print(f"\nInsertion du document: {doc_id}")
                # Passer explicitement le file_path à ainsert()
                await rag.ainsert(f.read(), file_paths=[file_path])
                print(f"Document '{doc_id}' inséré avec succès")
        except Exception as e:
            print(f"Erreur lors de l'insertion de {doc_id}: {str(e)}")
            raise  # Relancer l'exception pour faciliter le débogage


async def query_documents(rag, question, doc_ids=None):
    """
    Effectue des requêtes sur les documents chargés dans RAG.
    
    Args:
        rag: Instance de LightRAG
        question: Question à poser
        doc_ids: Liste des doc_ids à interroger (None pour tous les documents)
    """
    # Configuration des paramètres de requête
    params = {
        'naive': QueryParam(mode="naive"),
        'local': QueryParam(mode="local"),
        'global': QueryParam(mode="global"),
        'hybrid': QueryParam(mode="hybrid")
    }
    
    if doc_ids:
        print(f"\nRecherche dans les documents: {', '.join(doc_ids)}")
        for key in params:
            params[key] = QueryParam(mode=key, doc_ids=doc_ids)
    
    # Exécution des requêtes
    for mode, param in params.items():
        print(f"\n{'=' * 20}")
        print(f"Mode: {mode.upper()}")
        print(f"Question: {question}")
        print(f"{ '=' * 20}")
        try:
            response = await rag.aquery(question, param=param)
            if inspect.isasyncgen(response):
                await print_stream(response)
            else:
                print(response)
        except Exception as e:
            print(f"Erreur lors de la requête en mode {mode}: {str(e)}")


import httpx

async def infinity_embed(texts, embed_model, host, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": texts,
        "model": embed_model
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{host}/embeddings", headers=headers, json=payload)
        response.raise_for_status() # Lève une exception pour les codes d'état HTTP 4xx/5xx
        return response.json()["data"]


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: infinity_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"), # Le modèle peut être configuré via ENV
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:8000"), # URL de l'API Infinity
                api_key=os.getenv("EMBEDDING_API_KEY"), # Clé API pour l'API Infinity
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    # Vérification des variables d'environnement pour les modèles compatibles OpenAI
    if not os.getenv("LLM_MODEL") or not os.getenv("EMBEDDING_MODEL"):
        print(
            "Erreur: Les variables d'environnement LLM_MODEL et EMBEDDING_MODEL doivent être définies."
            "\nVeuillez définir ces variables avant d'exécuter le programme."
            "\nExemple: export LLM_MODEL='deepseek-chat'"
            "\nExemple: export EMBEDDING_MODEL='bge-m3:latest'"
        )
        return

    try:
        # Nettoyage des anciens fichiers
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
            "llm_response_cache.json", # Ajouté pour la cohérence
            "embedding_cache.json", # Ajouté pour la cohérence
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Suppression de l'ancien fichier: {file_path}")

        # Initialisation de LightRAG
        print("\nInitialisation de LightRAG...")
        rag = await initialize_rag()

        # Test de la fonction d'embedding
        test_text = ["Ceci est une chaîne de test pour l'embedding."]
        embedding = await rag.embedding_func(test_text)
        print("\nTest de la fonction d'embedding:")
        print(f"Texte de test: {test_text}")
        print(f"Dimension détectée: {embedding.shape[1]}")

        # Insertion des documents
        documents = {
            "dickens": "./book.txt",
            "le_petit_prince": "./book_2.txt"
        }
        await insert_documents(rag, documents)

        # Exemples de requêtes
        questions = [
            "Quelle est la signification du dessin du serpent boa que le narrateur a fait enfant ?",
        ]

        for question in questions:
            await query_documents(rag, question)

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        resp = await rag.aquery(
            "What are the top themes in this story?",
            param=QueryParam(mode="hybrid", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
