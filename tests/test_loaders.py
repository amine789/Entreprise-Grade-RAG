from enterprise_rag.loaders import load_directory_chunks, load_text_file_chunks


def test_load_text_file_chunks_splits_and_tags_metadata(tmp_path):
    path = tmp_path / "handbook.txt"
    path.write_text("word " * 1500)

    chunks = load_text_file_chunks(path, chunk_size=200, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(c["source"] == "handbook.txt" for c in chunks)
    assert [c["page"] for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c["content"] for c in chunks)


def test_load_text_file_chunks_single_chunk_for_short_text(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("just a few words")

    chunks = load_text_file_chunks(path)

    assert chunks == [
        {"content": "just a few words", "source": "short.txt", "page": 1}
    ]


def test_load_directory_chunks_reads_all_txt_files_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("second file")
    (tmp_path / "a.txt").write_text("first file")
    (tmp_path / "not_txt.md").write_text("ignored")

    chunks = load_directory_chunks(tmp_path)

    assert [c["source"] for c in chunks] == ["a.txt", "b.txt"]


def test_load_directory_chunks_empty_directory(tmp_path):
    assert load_directory_chunks(tmp_path) == []
