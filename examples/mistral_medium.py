import base64
from openai import OpenAI

# Configuration de l’API
API_KEY = "EMPTY"  # vLLM n’exige pas de clé par défaut, sinon votre clé
API_BASE = "http://localhost:8000/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

# Encodage de l’image en base64
with open("image_exemple.png", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# Préparation du message multimodal
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Décris cette image en détail."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
    }
]

# Appel de l’API compatible OpenAI (vLLM)
completion = client.chat.completions.create(
    model="mistralai/Mistral-Medium-3",  # Ou le nom exact du modèle déployé
    messages=messages,
    max_tokens=256,
    temperature=0.7
)

print(completion.choices[0].message.content)
