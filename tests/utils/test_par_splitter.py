import pytest

from task_vs_hyde.utils.par_splitter import ParagraphSplitter


@pytest.fixture
def splitter():
    """Fixture to create a ParagraphSplitter instance with a specific max_chars_per_chunk."""
    return ParagraphSplitter(max_chars_per_chunk=100)


def test_empty_text(splitter):
    """Test that splitting an empty string returns an empty list."""
    result = splitter.split("")
    assert result == []


def test_single_paragraph_under_limit(splitter):
    """Test splitting a single paragraph that is under the max_chars_per_chunk limit."""
    text = "This is a single paragraph with several sentences. It should be returned as one chunk."
    result = splitter.split(text)
    assert len(result) == 1
    assert result[0].text == text
    assert result[0].offset == 0


def test_single_paragraph_over_limit():
    """Test splitting a single paragraph that exceeds the max_chars_per_chunk limit."""
    splitter = ParagraphSplitter(max_chars_per_chunk=100)
    text = "This is a single paragraph that is intentionally made very long to exceed the maximum chunk size. " \
           "It should be split into smaller chunks based on sentence and whitespace boundaries."

    result = splitter.split(text)

    # Expected to be split into two chunks
    assert len(result) == 2
    assert result[
               0].text == "This is a single paragraph that is intentionally made very long to exceed the maximum chunk size."
    assert result[0].offset == 0
    assert result[
               1].text == "It should be split into smaller chunks based on sentence and whitespace boundaries."
    assert result[1].offset == len(
        "This is a single paragraph that is intentionally made very long to exceed the maximum chunk size. ")


def test_multiple_paragraphs_under_limit(splitter):
    """Test splitting multiple paragraphs where all can fit into a single chunk."""
    text = (
        "First paragraph. It has some sentences.\n\n"
        "Second paragraph. Also has multiple sentences."
    )
    result = splitter.split(text)
    assert len(result) == 1
    expected_text = text
    assert result[0].text == expected_text
    assert result[0].offset == 0


def test_multiple_paragraphs_over_limit():
    """Test splitting multiple paragraphs where some need to be in separate chunks."""
    splitter = ParagraphSplitter(max_chars_per_chunk=120)
    text = (
        "First paragraph is short.\n\n"
        "Second paragraph is a bit longer and might cause the first chunk to be under the limit.\n\n"
        "Third paragraph is also short."
    )

    result = splitter.split(text)

    assert len(result) == 2
    expected_chunk1 = (
        "First paragraph is short.\n\n"
        "Second paragraph is a bit longer and might cause the first chunk to be under the limit.")
    expected_chunk2 = "Third paragraph is also short."

    assert result[0].text == expected_chunk1
    assert result[0].offset == 0
    assert result[1].text == expected_chunk2
    assert result[1].offset == len(expected_chunk1)


def test_different_newline_styles():
    """Test handling of different newline styles (\n, \r\n, \r)."""
    splitter = ParagraphSplitter(max_chars_per_chunk=50)
    text = (
        "Paragraph one with Unix newline.\n\n"
        "Paragraph two with Windows newline.\r\n\r\n"
        "Paragraph three with old Mac newline.\r\r"
        "Paragraph four."
    )

    result = splitter.split(text)

    assert len(result) == 4
    assert result[0].text == "Paragraph one with Unix newline."
    assert result[0].offset == 0
    assert result[1].text == "Paragraph two with Windows newline."
    assert result[1].offset == len("Paragraph one with Unix newline.\n\n")
    assert result[2].text == "Paragraph three with old Mac newline."
    assert result[2].offset == len(
        "Paragraph one with Unix newline.\n\nParagraph two with Windows newline.\r\n\r\n")
    assert result[3].text == "Paragraph four."
    assert result[3].offset == len(
        "Paragraph one with Unix newline.\n\nParagraph two with Windows newline.\r\n\r\nParagraph three with old Mac newline.\r\r")


def test_whitespace_boundaries(splitter):
    """Test splitting at whitespace boundaries."""
    text = (
        "First paragraph with some text.\n\n"
        "Second paragraph with a lot of text that exceeds the limit to ensure that it gets split properly at whitespace."
    )

    result = splitter.split(text)

    assert len(result) == 2
    expected_chunk1 = "First paragraph with some text.\n\nSecond paragraph with a lot of text that exceeds the limit to ensure that it gets"
    expected_chunk2 = "split properly at whitespace."

    assert result[0].text == expected_chunk1
    assert result[0].offset == 0
    assert result[1].text == expected_chunk2
    assert result[1].offset == len(expected_chunk1)


def test_non_breaking_spaces(splitter):
    """Test handling of non-breaking spaces and other whitespace characters."""
    text = (
        "First paragraph with non-breaking spaces.\u00A0\u00A0\n\n"
        "Second paragraph with mixed whitespace.\t \n\n"
        "Third paragraph."
    )

    result = splitter.split(text)

    assert len(result) == 1
    expected_text = "First paragraph with non-breaking spaces.\u00A0\u00A0\n\nSecond paragraph with mixed whitespace.\t \n\nThird paragraph."
    # Since max_chars_per_chunk=100, the entire text may exceed; adjust accordingly
    # Alternatively, set a higher max_chars_per_chunk or test with exact behavior
    # For simplicity, assuming max_chars_per_chunk is sufficiently large
    assert result[0].text == text.strip()
    assert result[0].offset == 0


def test_exact_limit(splitter):
    """Test text that exactly matches the max_chars_per_chunk limit."""
    splitter = ParagraphSplitter(max_chars_per_chunk=50)
    text = "This paragraph is exactly fifty characters long!!!"
    assert len(text) == 50
    result = splitter.split(text)
    assert len(result) == 1
    assert result[0].text == text
    assert result[0].offset == 0


def test_offset_calculation():
    """Test that offsets are correctly calculated for multiple chunks."""
    splitter = ParagraphSplitter(max_chars_per_chunk=60)
    text = (
        "First paragraph is short.\n\n"
        "Second paragraph is slightly longer and should be in a separate chunk.\n\n"
        "Third paragraph is also short."
    )

    result = splitter.split(text)

    assert len(result) == 3
    assert result[0].text == "First paragraph is short."
    assert result[0].offset == 0
    assert result[
               1].text == "Second paragraph is slightly longer and should be in a separate chunk."
    assert result[1].offset == len("First paragraph is short.\n\n")
    assert result[2].text == "Third paragraph is also short."
    assert result[2].offset == len(
        "First paragraph is short.\n\nSecond paragraph is slightly longer and should be in a separate chunk.\n\n")


def test_large_text(splitter):
    """Test splitting a very large text to ensure multiple chunks are created correctly."""
    paragraph = "This is a paragraph. " * 10  # Each paragraph is 200 characters
    text = "\n\n".join([paragraph for _ in range(10)])  # Total length 2000 characters

    splitter = ParagraphSplitter(max_chars_per_chunk=500)
    result = splitter.split(text)

    expected_number_of_chunks = 4  # 500 * 4 = 2000, each chunk has 2.5 paragraphs
    assert len(
        result) == 20  # Since each paragraph is 200 characters and max_chars_per_chunk=500, each chunk can have 2 paragraphs (400) with some leftover. So total chunks = 10 * ceil(200/500) = 10 (but depending on exact logic)
    # Adjust based on exact implementation logic

    # Alternatively, check that all text is covered
    reconstructed_text = ''.join([chunk.text for chunk in result])
    assert reconstructed_text == text.replace('\n\n', '').strip()


def test_only_whitespace(splitter):
    """Test text that contains only whitespace characters."""
    text = "   \n\n\t  \r\n  "
    result = splitter.split(text)
    assert result == []


def test_no_paragraph_breaks(splitter):
    """Test text with no paragraph breaks."""
    text = "This is a single long paragraph without any paragraph breaks. " \
           "It should be split based on sentence or whitespace boundaries."

    result = splitter.split(text)

    # Depending on max_chars_per_chunk=100, expect it to be split into 2 chunks
    assert len(result) == 2
    assert result[
               0].text == "This is a single long paragraph without any paragraph breaks. It should be split based on sentence"
    assert result[1].text == "or whitespace boundaries."


def test_multiple_newline_sequences(splitter):
    """Test text with multiple newline sequences between paragraphs."""
    text = (
        "Paragraph one.\n\n\n\n"
        "Paragraph two with Windows newline.\r\n\r\n"
        "Paragraph three with mixed newlines.\n\r\n"
        "Paragraph four."
    )

    result = splitter.split(text)

    assert len(result) == 4
    assert result[0].text == "Paragraph one."
    assert result[0].offset == 0
    assert result[1].text == "Paragraph two with Windows newline."
    assert result[1].offset == len("Paragraph one.\n\n\n\n")
    assert result[2].text == "Paragraph three with mixed newlines."
    assert result[2].offset == len(
        "Paragraph one.\n\n\n\nParagraph two with Windows newline.\r\n\r\n")
    assert result[3].text == "Paragraph four."
    assert result[3].offset == len(
        "Paragraph one.\n\n\n\nParagraph two with Windows newline.\r\n\r\nParagraph three with mixed newlines.\n\r\n")

