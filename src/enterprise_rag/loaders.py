from pathlib import Path

from langchain_text_splitters import TokenTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_text_file_chunks(
    path: Path, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Load a text file and split it into token-aware chunks.

    Returns chunks shaped for ingest_documents: each has "content", "source"
    (the file name), and "page" (1-indexed chunk number within the file).
    """
    text = path.read_text(encoding="utf-8")
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        {"content": chunk, "source": path.name, "page": i + 1}
        for i, chunk in enumerate(splitter.split_text(text))
    ]


def load_directory_chunks(
    directory: Path, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Load and chunk every .txt file in a directory."""
    chunks = []
    for path in sorted(directory.glob("*.txt")):
        chunks.extend(load_text_file_chunks(path, chunk_size, chunk_overlap))
    return chunks
