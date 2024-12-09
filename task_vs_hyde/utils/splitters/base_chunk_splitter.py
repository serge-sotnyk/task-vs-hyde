from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TextChunk:
    """
    Represents a chunk of text along with its offset in the original text.
    """
    text: str
    offset: int


class BaseChunkSplitter(ABC):
    """Base class for text chunk splitters"""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars

    @abstractmethod
    def split(self, text: str) -> list[TextChunk]:
        """Split text into chunks"""
        raise NotImplementedError()


class BaseRecursiveChunkSplitter(BaseChunkSplitter):
    def __init__(self, max_chars: int):
        super().__init__(max_chars)
        self.next_splitter: BaseChunkSplitter | None = None

    def split(self, text: str) -> list[TextChunk]:
        starts = self.minichunks_starts_positions(text)
        ends = starts[1:] + [len(text)]
        chunks = []
        current_offset = 0
        for start, end in zip(starts, ends):
            if end - current_offset > self.max_chars:
                new_chunk = TextChunk(text[current_offset:start], current_offset)
                chunks.append(new_chunk)
                current_offset = start
                if end - start > self.max_chars:
                    new_chunks = self.next_splitter.split(text[start:end])
                    for chunk in new_chunks:
                        chunk.offset += start
                    chunks.extend(new_chunks)
                    current_offset = end
        if current_offset < len(text):
            chunks.append(TextChunk(text[current_offset:], current_offset))
        # remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.text]
        return chunks

    @abstractmethod
    def minichunks_starts_positions(self, text: str) -> list[int]:
        """Split text into chunks and return their positions in the original text"""
        raise NotImplementedError()
