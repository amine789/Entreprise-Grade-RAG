import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json

import numpy as np
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
import faiss

from enterprise_rag.pipeline import handle_query

TIME_SENSITIVE_KEYWORDS = [
    "today", "tonight", "now", "currently", "current",
    "latest", "recent", "recently", "right now",
    "at the moment", "at present", "as of now",
    "this week", "this month", "this year",
    "this quarter", "this season", "this morning",
    "this afternoon", "this evening", "this weekend",
    "yesterday", "tomorrow", "last week", "last month",
    "last year", "upcoming", "live", "breaking",
    "just happened", "what time", "what day", "what date",
    "happening now", "events today", "news today",
    "news this week", "stock price", "share price",
    "weather", "forecast", "temperature",
    "real-time", "realtime", "schedule today",
    "outage", "down right now",
]


def is_time_sensitive(question: str) -> bool:
    question_lower = question.lower()
    return any(
        keyword in question_lower
        for keyword in TIME_SENSITIVE_KEYWORDS
    )


class SemanticCaching:
    def __init__(
        self,
        json_file='cache.json',
        threshold=0.2,
        clear_on_init=False
    ):
        self.embedding_dim = 768
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.euclidean_threshold = threshold
        self.json_file = json_file
        self.encoder = SentenceTransformer(
            'nomic-ai/nomic-embed-text-v1.5',
            trust_remote_code=True
        )
        if clear_on_init:
            self.clear_cache()
        else:
            self.load_cache()

    def load_cache(self):
        try:
            with open(self.json_file, 'r') as f:
                self.cache = json.load(f)
            if self.cache['embeddings']:
                embeddings = np.array(
                    self.cache['embeddings'],
                    dtype=np.float32
                )
                self.index.add(embeddings)
        except FileNotFoundError:
            self.cache = {
                'questions': [],
                'embeddings': [],
                'response_text': []
            }

    def check_cache(self, question: str):
        embedding = self.encoder.encode(
            [question],
            normalize_embeddings=True
        )
        if self.index.ntotal == 0:
            return False, None, embedding, None, None
        D, I = self.index.search(embedding, 1)
        if I[0][0] != -1 and D[0][0] <= self.euclidean_threshold:
            row_id = int(I[0][0])
            similarity = float(1.0 - D[0][0])
            return (
                True,
                self.cache['response_text'][row_id],
                embedding,
                similarity,
                row_id
            )
        return False, None, embedding, None, None

    def add_to_cache(
        self, question: str, answer: str, embedding
    ):
        self.cache['questions'].append(question)
        self.cache['embeddings'].append(
            embedding[0].tolist()
        )
        self.cache['response_text'].append(answer)
        self.index.add(embedding)
        self.save_cache()

    def save_cache(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.cache, f)

    def clear_cache(self):
        self.cache = {
            'questions': [],
            'embeddings': [],
            'response_text': []
        }
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.save_cache()


async def ask(cache: SemanticCaching, qdrant: AsyncQdrantClient, question: str) -> str:
    if not is_time_sensitive(question):
        hit, cached_answer, embedding, _, _ = cache.check_cache(question)
        if hit:
            return cached_answer
    else:
        embedding = cache.encoder.encode([question], normalize_embeddings=True)

    answer = await handle_query(qdrant, question)
    cache.add_to_cache(question, answer, embedding)
    return answer
