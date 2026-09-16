import torch
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

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

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)


def get_text_embeddings(text, tokenizer=tokenizer, model=model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].detach().numpy()
