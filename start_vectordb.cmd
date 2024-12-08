docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 -v "%cd%\postgres-data:/var/lib/postgresql/data" pgvector/pgvector:pg17
