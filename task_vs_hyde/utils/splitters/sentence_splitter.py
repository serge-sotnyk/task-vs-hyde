import re

from task_vs_hyde.utils.splitters.base_chunk_splitter import BaseRecursiveChunkSplitter
from task_vs_hyde.utils.splitters.word_splitter import WordSplitter


class SentenceSplitter(BaseRecursiveChunkSplitter):
    """Splits text into chunks by sentence boundaries"""

    def __init__(self, max_chars: int):
        super().__init__(max_chars)
        self.next_splitter = WordSplitter(max_chars)
        # Compile the regex pattern for sentence boundaries
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def minichunks_starts_positions(self, text: str) -> list[int]:
        """
        Returns list of positions where sentences start in the text using regex.
        Considers punctuation as sentence boundaries.
        """
        # Find all sentence boundaries
        matches = list(self.sentence_pattern.finditer(text))
        
        # Get positions of sentence starts and add 0 as the first position
        positions = [0] + [match.end() for match in matches]
        
        # Remove duplicates and sort
        return sorted(list(set(positions)))