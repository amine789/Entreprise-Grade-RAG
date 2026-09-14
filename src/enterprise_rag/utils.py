import torch
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)


def get_text_embeddings(text, tokenizer=tokenizer, model=model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].detach().numpy()
