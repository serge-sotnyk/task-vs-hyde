import re
from task_vs_hyde.utils.splitters.base_chunk_splitter import BaseRecursiveChunkSplitter
from task_vs_hyde.utils.splitters.char_splitter import CharacterSplitter


class WordSplitter(BaseRecursiveChunkSplitter):
    """Splits text into chunks by word boundaries"""

    def __init__(self, max_chars: int):
        super().__init__(max_chars)
        self.next_splitter = CharacterSplitter(max_chars)
        self.word_pattern = re.compile(r'\b\w')  # Compile the regex pattern

    def minichunks_starts_positions(self, text: str) -> list[int]:
        """
        Returns list of positions where words start in the text using regex.
        Considers whitespace and punctuation as word boundaries.
        """
        # Find all word boundaries
        matches = list(self.word_pattern.finditer(text))
        
        # Get positions of word starts and add 0 as the first position
        positions = [0] + [match.start() for match in matches]
        
        # Remove duplicates and sort
        return sorted(list(set(positions)))