import time
from textwrap import dedent

from nomic import embed


def get_embeddings(text: str | list[str]) -> list[list[float]]:
    # https://www.nomic.ai/blog/posts/local-nomic-embed
    if isinstance(text, str):
        text = [text]
    output = embed.text(
        texts=text,
        model='nomic-embed-text-v1.5',
        task_type="search_document",
        inference_mode='local',
        device='gpu',
        dimensionality=768,
    )
    embs = output['embeddings']
    return embs


def main():
    text = "Hello world"
    res = get_embeddings(text)
    print(res)
    print("Embedding size:", len(res))

    # Parallel embeddings encoding
    texts = [
        "search_query: what is a hello world?",
        dedent("""
        search_document: A "Hello, World!" program is generally a simple computer program that
        emits (or displays) to the screen (often the console) a message similar to "Hello,
        World!". A small piece of code in most general-purpose programming languages, this program
        is used to illustrate a language's basic syntax. A "Hello, World!" program is often the
        first written by a student of a new programming language,[1] but such a program can also
        be used as a sanity check to ensure that the computer software intended to compile or run
        source code is correctly installed, and that its operator understands how to use it.
        """)
    ]
    for n in range(100):
        start = time.perf_counter()
        _ = get_embeddings(texts)
        end = time.perf_counter()
        print(f"{n}: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
