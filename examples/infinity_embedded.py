infinity_embedder = InfinityEmbeddingFunc(endpoint_url="http://localhost:8000/embed", embedding_dim=1024)

rag = LightRAG(
    working_dir=WORKING_DIR,
    embedding_func=infinity_embedder,
    llm_model_func=...,
)


------


from lightrag.utils import EmbeddingFunc
import numpy as np
import httpx  # Asynchrone

class InfinityEmbeddingFunc(EmbeddingFunc):
    # Variables en dur pour tests
    ENDPOINT_URL = "http://localhost:8000/embed"
    API_KEY = "sk-demo-1234567890"

    def __init__(self, embedding_dim=1024, max_token_size=8192):
        super().__init__(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=self.func
        )

    async def func(self, texts: list[str]) -> np.ndarray:
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"texts": texts}
        async with httpx.AsyncClient() as client:
            response = await client.post(self.ENDPOINT_URL, json=payload, headers=headers)
            response.raise_for_status()
            vectors = response.json()["embeddings"]  # Adapter selon la réponse exacte de ton API
            arr = np.array(vectors, dtype=np.float32)
            assert arr.shape == (len(texts), self.embedding_dim), f"Expected {(len(texts), self.embedding_dim)}, got {arr.shape}"
            return arr


---



from lightrag.utils import EmbeddingFunc
import numpy as np
import os
import httpx  # Asynchrone et moderne

class InfinityEmbeddingFunc(EmbeddingFunc):
    # Variables en dur pour test/démo (à adapter selon ton cas réel)
    API_URL = "https://dmn-ap261!ilab.cloud.echonet/ULma/meet.../ULm-serv-infinity/..."  # Mets l’URL complète ici
    MODEL = "multilingual-e5-large"
    ENCODING_FORMAT = "float"
    MODALITY = "text"
    USER = "string"  # Peut être adapté ou rendu dynamique

    def __init__(self, embedding_dim=1024, max_token_size=8192):
        super().__init__(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=self.func
        )

    async def func(self, texts: list[str]) -> np.ndarray:
        api_key = os.getenv("LMMAAS_API_KEY") or "sk-demo-1234567890"  # fallback pour test
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": self.MODEL,
            "encoding_format": self.ENCODING_FORMAT,
            "user": self.USER,
            "input": texts,
            "modality": self.MODALITY
        }
        # Appel asynchrone
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(self.API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            # Adapter la clé selon la vraie réponse de ton API
            vectors = response_data["embeddings"]
            arr = np.array(vectors, dtype=np.float32)
            assert arr.shape == (len(texts), self.embedding_dim), f"Expected {(len(texts), self.embedding_dim)}, got {arr.shape}"
            return arr
