from task_vs_hyde.utils.splitters.base_chunk_splitter import BaseChunkSplitter, TextChunk


class CharacterSplitter(BaseChunkSplitter):
    """Splits text into chunks of fixed size"""

    def split(self, text: str, offset: int = 0) -> list[TextChunk]:
        chunks = []
        for i in range(0, len(text), self.max_chars):
            chunk = text[i:i + self.max_chars]
            chunks.append(TextChunk(chunk, offset + i))
        return chunks