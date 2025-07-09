import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug

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
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
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


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


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
        print(f"{'=' * 20}")
        try:
            response = await rag.aquery(question, param=param)
            print(response)
        except Exception as e:
            print(f"Erreur lors de la requête en mode {mode}: {str(e)}")


async def main():
    # Vérification de la clé API OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Erreur: La variable d'environnement OPENAI_API_KEY n'est pas définie."
            "\nVeuillez définir cette variable avant d'exécuter le programme."
            "\nExemple: export OPENAI_API_KEY='votre-clef-api'"
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
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Suppression de l'ancien fichier: {file_path}")

        # Initialisation de RAG
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
            # Requête sur tous les documents
            await query_documents(rag, question)
            
            # Requête sur un document spécifique
            # await query_documents(rag, question, doc_ids=["livre1"])

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



------





from lightrag.utils import EmbeddingFunc
import numpy as np
import httpx  # Pour les appels HTTP asynchrones

class InfinityEmbeddingFunc(EmbeddingFunc):
    API_URL = "https://ton-endpoint-infinity.com/embed"  # Mets ici ton URL complète
    API_KEY = "sk-demo-1234567890"  # Mets ici ta clé API Infinity

    def __init__(self, embedding_dim=1024, max_token_size=8192):
        super().__init__(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=self.func
        )

    async def func(self, texts: list[str]) -> np.ndarray:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        results = []
        async with httpx.AsyncClient(verify=False) as client:
            for text in texts:
                payload = {
                    "model": "multilingual-e5-large",
                    "encoding_format": "float",
                    "user": "string",
                    "input": [text],  # Adapter à "input": text si l'API refuse la liste
                    "modality": "text"
                }
                response = await client.post(self.API_URL, json=payload, headers=headers)
                response.raise_for_status()
                response_data = response.json()
                # Adapter la clé selon la vraie réponse de ton API
                vec = response_data["embeddings"]
                arr = np.array(vec, dtype=np.float32)
                # Si l'API retourne un embedding 1D, reshape
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                results.append(arr[0])
        all_embeddings = np.vstack(results)
        assert all_embeddings.shape == (len(texts), self.embedding_dim), f"Expected {(len(texts), self.embedding_dim)}, got {all_embeddings.shape}"
        return all_embeddings



----

import asyncio
import httpx
import json

# Paramètres à adapter
API_URL = "https://ton-endpoint-infinity.com/embed"   # Mets ici ton URL complète
API_KEY = "sk-demo-1234567890"                        # Mets ici ta clé API Infinity

async def test_infinity():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "multilingual-e5-large",
        "encoding_format": "float",
        "user": "string",
        "input": ["Bonjour, ceci est un test depuis LightRAG!"],
        "modality": "text"
    }
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(API_URL, json=payload, headers=headers)
            print("Status code:", response.status_code)
            print("Réponse brute:", response.text)
            response.raise_for_status()
            response_data = response.json()
            print("Réponse JSON:", json.dumps(response_data, indent=2))
        except Exception as e:
            print("Erreur lors de l'appel au endpoint Infinity:", e)

if __name__ == "__main__":
    asyncio.run(test_infinity())


--- 

1
