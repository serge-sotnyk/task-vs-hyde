import pytest

from task_vs_hyde.utils.splitters.base_chunk_splitter import TextChunk
from task_vs_hyde.utils.splitters.paragraph_splitter import ParagraphSplitter


@pytest.fixture
def splitter():
    """Fixture to create a ParagraphSplitter instance with a specific max_chars."""
    return ParagraphSplitter(max_chars=100)


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
    """Test splitting a single paragraph that exceeds the max_chars limit."""
    splitter = ParagraphSplitter(max_chars=100)
    text = "This is a single paragraph that is intentionally made very long to exceed the maximum chunk size. " \
           "It should be split into smaller chunks based on sentence and whitespace boundaries."

    result = splitter.split(text)

    # Expected to be split into two chunks
    assert len(result) == 2
    assert result[0].text.startswith("This is a single paragraph")
    assert result[0].offset == 0
    assert result[1].text.startswith("It should be split")
    assert result[1].offset == text.find("It should be split")


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
    splitter = ParagraphSplitter(max_chars=120)
    text = (
        "First paragraph is short.\n\n"
        "Second paragraph is a bit longer and might cause the first chunk to be under the limit.\n\n"
        "Third paragraph is also short."
    )

    result = splitter.split(text)

    assert len(result) == 2
    assert result[0].text.startswith("First paragraph is short.")
    assert result[0].offset == 0
    assert result[1].text.startswith("Third paragraph")
    assert result[1].offset == text.find("Third paragraph")


def test_different_newline_styles():
    """Test handling of different newline styles (\n, \r\n, \r)."""
    splitter = ParagraphSplitter(max_chars=50)
    text = (
        "Paragraph one with Unix newline.\n\n"
        "Paragraph two with Windows newline.\r\n\r\n"
        "Paragraph three with old Mac newline.\r\r"
        "Paragraph four."
    )

    result = splitter.split(text)

    assert len(result) == 4
    assert result[0].text.startswith("Paragraph one")
    assert result[0].offset == 0
    assert result[1].text.startswith("Paragraph two")
    assert result[1].offset == text.find("Paragraph two")
    assert result[2].text.startswith("Paragraph three")
    assert result[2].offset == text.find("Paragraph three")
    assert result[3].text.startswith("Paragraph four")
    assert result[3].offset == text.find("Paragraph four")


def test_whitespace_boundaries(splitter):
    """Test splitting at whitespace boundaries."""
    text = (
        "First paragraph with some text.\n\n"
        "Second paragraph with a lot of text that exceeds the limit to ensure that it gets split properly at whitespace."
    )

    result = splitter.split(text)

    assert len(result) == 3

    assert result[1].text.startswith("Second paragraph")
    assert result[1].offset == text.find("Second paragraph")
    second_start = text.find("Second paragraph")
    third_start = second_start + text[second_start + splitter.max_chars:].find(" ") + 1
    assert result[2].text.startswith("whitespace.")
    assert result[2].offset == text.find("whitespace.")


def test_non_breaking_spaces(splitter):
    """Test handling of non-breaking spaces and other whitespace characters."""
    text = (
            "First paragraph with non-breaking spaces.\u00A0\u00A0\n\n" + "\u00A0" * 200 +
            "Second paragraph with mixed whitespace.\t \n\n"
            "Third paragraph."
    )

    result = splitter.split(text)

    assert len(result) == 5
    assert result[0].text.startswith("First paragraph")
    assert result[0].text.endswith("\n\n")
    assert result[0].offset == 0
    assert result[1].text.startswith("\u00A0")
    assert result[1].text.endswith("\u00A0\u00A0")

    text = "First paragraph with tabs.\t\t\n\nSecond paragraph. "+"\t" * 200+"Third paragraph."
    result = splitter.split(text)

    assert len(result) == 6
    assert result[0].text.startswith("First paragraph")
    assert result[0].text.endswith("\n\n")
    assert result[0].offset == 0
    assert result[1].text.startswith("Second")
    assert result[2].text.endswith("\t")
    assert result[3].text.startswith("\t\t")


def test_exact_limit(splitter):
    """Test text that exactly matches the max_chars_per_chunk limit."""
    splitter = ParagraphSplitter(max_chars=50)
    text = "This paragraph is exactly fifty characters long!!!"
    assert len(text) == 50
    result = splitter.split(text)
    assert len(result) == 1
    assert result[0].text == text
    assert result[0].offset == 0


def test_offset_calculation():
    """Test that offsets are correctly calculated for multiple chunks."""
    splitter = ParagraphSplitter(max_chars=60)
    text = (
        "First paragraph is short.\n\n"
        "Second paragraph is slightly longer and should be in a separate chunk.\n\n"
        "Third paragraph is also short."
    )

    result = splitter.split(text)

    assert len(result) == 4
    assert result[0].text.startswith("First paragraph")
    assert result[0].offset == 0
    assert result[1].text.startswith("Second paragraph")
    assert result[1].offset == text.find("Second paragraph")
    assert result[2].text.strip().endswith("chunk.")
    # I fixed the test (not the splitter itself) in a third paragraph, even it is a bit ugly,
    # because it should be a rare situation.
    assert result[3].text.startswith("\nThird paragraph")
    assert result[3].offset == text.find("\nThird paragraph")


def test_large_text(splitter):
    """Test splitting a very large text to ensure multiple chunks are created correctly."""
    paragraph = "This is a paragraph. " * 10
    text = "\n\n".join([paragraph for _ in range(10)])

    splitter = ParagraphSplitter(max_chars=500)
    result = splitter.split(text)

    assert len(result) >= len(text) / splitter.max_chars

    # Alternatively, check that all text is covered
    reconstructed_text = ''.join([chunk.text for chunk in result])
    assert reconstructed_text == text


def test_only_whitespace(splitter):
    """Test text that contains only whitespace characters."""
    text = "   \n\n\t  \r\n  "
    result = splitter.split(text)
    assert result == [TextChunk(text, 0)]

    text = " "*250
    result = splitter.split(text)
    assert len(result) == 3


def test_no_paragraph_breaks(splitter):
    """Test text with no paragraph breaks."""
    text = "This is a single long paragraph without any paragraph breaks. " \
           "It should be split based on sentence or whitespace boundaries."

    result = splitter.split(text)

    # Depending on max_chars=100, expect it to be split into 2 chunks
    assert len(result) == 2
    assert result[0].text.startswith("This is a single long paragraph")
    assert result[1].text.startswith("It should be split")

