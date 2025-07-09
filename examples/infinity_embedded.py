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
import httpx  # Pour les appels HTTP asynchrones

class InfinityEmbeddingFunc(EmbeddingFunc):
    # Variables en dur pour test/démo
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
        payload = {
            "model": "multilingual-e5-large",
            "encoding_format": "float",
            "user": "string",
            "input": texts,
            "modality": "text"
        }
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(self.API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            # Adapter la clé selon la vraie réponse de ton API
            vectors = response_data["embeddings"]
            arr = np.array(vectors, dtype=np.float32)
            # Correction : toujours retourner un tableau 2D
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            assert arr.shape == (len(texts), self.embedding_dim), f"Expected {(len(texts), self.embedding_dim)}, got {arr.shape}"
            return arr



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
