from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer(
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    trust_remote_code=True,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",  # Optional
    },
)
model.max_seq_length = 512

sentences = ["This is an example sentence", "Each sentence is converted"]
prompt = "Instruct: Classifying the category of french news.\nQuery:"
embeddings = model.encode(
    sentences,
    prompt=prompt,
    normalize_embeddings=True,
    batch_size=256,
    show_progress_bar=True,
)
print(embeddings)
