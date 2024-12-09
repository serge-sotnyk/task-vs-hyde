import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    """
    Represents a chunk of text along with its offset in the original text.
    """
    text: str
    offset: int


class ParagraphSplitter:
    """
    A text splitter that attempts to divide the text into chunks by grouping multiple paragraphs
    without exceeding a specified character limit. If a single paragraph exceeds the limit,
    it falls back to splitting by sentences, whitespace, and finally by character length.
    """

    def __init__(self, max_chars_per_chunk: int = 1024):
        """
        Initializes the ParagraphSplitter.

        :param max_chars_per_chunk: The maximum number of characters allowed per chunk.
        """
        self.max_chars_per_chunk = max_chars_per_chunk
        # Precompile the sentence boundary regex for efficiency
        self.sentence_boundary_pattern = re.compile(r'[.?!](?=\s|$)')

    def _find_split_index(self, text: str) -> int:
        """
        Determines the optimal index to split the text based on defined priorities.

        Priorities:
        1) Split at sentence boundary (.!? followed by a space or the end of the string).
        2) Split at any whitespace character.
        3) If none are found, split strictly by max_chars_per_chunk.

        :param text: The text segment to evaluate for splitting.
        :return: The index at which to split the text.
        """
        limit = self.max_chars_per_chunk
        if len(text) <= limit:
            return len(text)

        # Attempt to find a sentence boundary
        sent_break_candidates = list(self.sentence_boundary_pattern.finditer(text[:limit]))
        if sent_break_candidates:
            return sent_break_candidates[-1].end()

        # Attempt to find any whitespace boundary, going backwards from the limit
        whitespace_pos = -1
        for i in range(limit - 1, -1, -1):
            if text[i].isspace():
                whitespace_pos = i
                break
        if whitespace_pos != -1 and whitespace_pos > 0:
            return whitespace_pos

        # No suitable boundary found - fallback to max_chars_per_chunk
        return limit

    def _split_paragraph(self, paragraph: str, start_offset: int) -> list[TextChunk]:
        """
        Splits a single paragraph into smaller chunks if it exceeds max_chars_per_chunk.

        :param paragraph: The paragraph text to split.
        :param start_offset: The starting offset of the paragraph in the original text.
        :return: A list of TextChunk instances for the paragraph.
        """
        chunks = []
        remaining_text = paragraph
        current_offset = start_offset

        while remaining_text:
            idx = self._find_split_index(remaining_text)
            chunk_raw = remaining_text[:idx]
            # Calculate leading whitespace to determine accurate offset
            leading_ws_count = 0
            while leading_ws_count < len(chunk_raw) and chunk_raw[leading_ws_count].isspace():
                leading_ws_count += 1

            chunk_text = chunk_raw.strip()
            chunk_offset = current_offset + leading_ws_count

            if chunk_text:  # Avoid adding empty chunks
                chunks.append(TextChunk(text=chunk_text, offset=chunk_offset))

            # Update the current offset by the index where the split occurred
            current_offset += idx
            remaining_text = remaining_text[idx:]

        return chunks

    def split(self, text: str) -> list[TextChunk]:
        """
        Splits the provided text into chunks not exceeding max_chars_per_chunk by grouping
        multiple paragraphs. If a single paragraph exceeds the limit, it is further split
        into smaller chunks based on sentences and whitespace.

        :param text: The original text to be split.
        :return: A list of TextChunk instances containing each chunk and its offset.
        """
        # Identify paragraph boundaries using splitlines with keeping line endings
        # Paragraphs are separated by two or more consecutive newline characters
        paragraph_pattern = re.compile(r'(?:\r?\n){2,}')
        paragraphs = list(paragraph_pattern.split(text))

        # Find the start index of each paragraph
        paragraph_boundaries = [match.start() for match in paragraph_pattern.finditer(text)]
        paragraphs_with_offsets = []
        current_offset = 0

        for i, paragraph in enumerate(paragraphs):
            # Skip empty paragraphs
            if not paragraph.strip():
                continue

            paragraphs_with_offsets.append((paragraph, current_offset))
            # Update current_offset by adding the length of the paragraph and the delimiter
            if i < len(paragraph_boundaries):
                delimiter_length = paragraph_boundaries[i] - (current_offset + len(paragraph))
                current_offset += len(paragraph) + delimiter_length
            else:
                current_offset += len(paragraph)

        chunks: list[TextChunk] = []
        current_chunk = ""
        chunk_start_offset: int | None = None

        for paragraph, para_offset in paragraphs_with_offsets:
            para_length = len(paragraph)
            if len(current_chunk) + para_length + 2 <= self.max_chars_per_chunk:
                # +2 accounts for the paragraph delimiter (e.g., two newlines)
                if not current_chunk:
                    chunk_start_offset = para_offset
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    # Remove the trailing newlines
                    chunk_text = current_chunk.rstrip('\n')
                    chunks.append(TextChunk(
                        text=chunk_text,
                        offset=chunk_start_offset if chunk_start_offset is not None else 0
                    ))
                    current_chunk = ""
                    chunk_start_offset = None

                if para_length + 2 > self.max_chars_per_chunk:
                    # Paragraph itself exceeds the limit, need to split it
                    split_chunks = self._split_paragraph(paragraph, para_offset)
                    chunks.extend(split_chunks)
                else:
                    # Start a new chunk with the current paragraph
                    current_chunk = paragraph + "\n\n"
                    chunk_start_offset = para_offset

        # Append any remaining chunk
        if current_chunk:
            chunk_text = current_chunk.rstrip('\n')
            chunks.append(TextChunk(
                text=chunk_text,
                offset=chunk_start_offset if chunk_start_offset is not None else 0
            ))

        return chunks
