# task-vs-hyde
Comparison for task-prefix-embeddings with HyDE approach

[Nomad embeddings](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) has the
prefixes that could be used as an 
[HyDE](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
approach replacement. And here we check the performance of the two approaches.

## Experiment's schema:

What we use:
- NOMIC Embeddings (~~through Ollama~~ - locally)
- deepseek-v2:latest as a LLM
- Groq llama 3.3 70b as an option for LLM
- docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 -v d:\work\pg_vector\postgres-data:/var/lib/postgresql/data pgvector/pgvector:pg17 as a vector storage
1. Determine what and how to calculate metrics.
2. Write an index builder (with a prefix and without - 2 modes)
3. Write a simple RAG (with a prefix, without, and with a HyDE mode)
4. Calculate metrics.