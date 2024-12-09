from itertools import accumulate

from task_vs_hyde.utils.splitters.base_chunk_splitter import BaseRecursiveChunkSplitter
from task_vs_hyde.utils.splitters.sentence_splitter import SentenceSplitter


class ParagraphSplitter(BaseRecursiveChunkSplitter):
    """Splits text into chunks by sentence boundaries"""

    def __init__(self, max_chars: int):
        super().__init__(max_chars)
        self.next_splitter = SentenceSplitter(max_chars) 

    def minichunks_starts_positions(self, text: str) -> list[int]:
        """
        Returns list of positions where paragraphs starts.
        """
        paragraphs = text.splitlines(keepends=True)

        positions = list(accumulate(len(paragraph) for paragraph in paragraphs))
        positions = [0] + positions[:-1]

        return positions
