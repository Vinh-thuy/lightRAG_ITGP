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
