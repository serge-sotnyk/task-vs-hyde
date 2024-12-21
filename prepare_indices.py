# Utility that prepares 2 variants of vector indices for experiment:
# - with prefix 'search_document'
# - without prefix (for vanilla and HyDE searching)
import os
from pathlib import Path
from typing import List

import dotenv
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Text,
    Index,
    select,
    text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import event
from pgvector.sqlalchemy import Vector
from pgvector.psycopg import register_vector
from nomic import embed
import psycopg

from task_vs_hyde.ds import DatasetItem
from task_vs_hyde.ds.reader import read_ds

# Load environment variables from .env
dotenv.load_dotenv(override=True)

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "54320")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "vector_db")  # Changed to 'vector_db'


# Function to ensure the database exists
def ensure_database_exists():
    """
    Connect to the default 'postgres' database and create the target database
    if it does not exist.
    """
    default_db = "postgres"
    default_database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{default_db}"
    target_database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Create an engine for the default database with AUTOCOMMIT
    default_engine = create_engine(
        default_database_url,
        isolation_level="AUTOCOMMIT",  # Set isolation level to AUTOCOMMIT
        echo=False
    )

    try:
        # Connect to the default database
        with default_engine.connect() as conn:
            # Check if the target database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": DB_NAME}
            )
            exists = result.scalar() is not None

            if not exists:
                print(f"Database '{DB_NAME}' does not exist. Creating it...")
                try:
                    conn.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
                    print(f"Database '{DB_NAME}' created successfully.")
                except Exception as e:
                    print(f"Failed to create database '{DB_NAME}': {e}")
                    raise
            else:
                print(f"Database '{DB_NAME}' already exists.")
    except Exception as e:
        print(f"Error ensuring database exists: {e}")
        raise
    finally:
        # Dispose the default engine as it's no longer needed
        default_engine.dispose()


# Ensure the target database exists before proceeding
ensure_database_exists()

# Now create the main engine for the target database
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)


# Register the vector type with Psycopg2
@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    register_vector(dbapi_connection)


# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Define the model for the vanilla index
class VanillaItem(Base):
    __tablename__ = "vanilla_items"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(768))  # Dimension of NOMIC embeddings

    __table_args__ = (
        Index(
            'idx_vanilla_embedding',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        ),
    )


# Define the model for the prefixed index
class PrefixedItem(Base):
    __tablename__ = "prefixed_items"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(768))  # Dimension of NOMIC embeddings

    __table_args__ = (
        Index(
            'idx_prefixed_embedding',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        ),
    )


def get_embeddings(text: str | List[str]) -> List[List[float]]:
    """
    Retrieve embeddings for a single text or a list of texts.

    :param text: A string or a list of strings to embed.
    :return: A list of embeddings.
    """
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


def clear_db():
    """
    Clear the database by dropping existing tables and extensions,
    then recreate them.
    """
    Base.metadata.drop_all(bind=engine)
    # Recreate the pgvector extension
    with engine.connect() as conn:
        conn.execute(text('DROP EXTENSION IF EXISTS vector'))
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    Base.metadata.create_all(bind=engine)
    print("Database has been cleared and prepared.")


def create_vanilla_index(ds: List[DatasetItem]):
    """
    Create the vanilla index without prefix.

    :param ds: List of dataset items.
    """
    session = SessionLocal()
    try:
        texts = [item.text for item in ds]
        print("Fetching embeddings for the vanilla index...")
        embeddings = get_embeddings(texts)
        print("Inserting data into the vanilla index...")
        for item, emb in zip(ds, embeddings):
            db_item = VanillaItem(text=item.text, embedding=emb)
            session.add(db_item)
        session.commit()
        print("Vanilla index created successfully.")
    except Exception as e:
        session.rollback()
        print(f"Error creating the vanilla index: {e}")
    finally:
        session.close()


def create_prefixed_index(ds: List[DatasetItem]):
    """
    Create the prefixed index with 'search_document' prefix.

    :param ds: List of dataset items.
    """
    session = SessionLocal()
    try:
        prefixed_texts = [f"search_document: {item.text}" for item in ds]
        print("Fetching embeddings for the prefixed index...")
        embeddings = get_embeddings(prefixed_texts)
        print("Inserting data into the prefixed index...")
        for item, emb in zip(ds, embeddings):
            db_item = PrefixedItem(text=f"search_document: {item.text}", embedding=emb)
            session.add(db_item)
        session.commit()
        print("Prefixed index created successfully.")
    except Exception as e:
        session.rollback()
        print(f"Error creating the prefixed index: {e}")
    finally:
        session.close()


def main():
    """
    Main function to execute the workflow:
    1. Clear the database.
    2. Read the dataset.
    3. Create both vanilla and prefixed indices.
    """
    clear_db()
    ds_root = Path(__file__).parent / "ds" / "manuals" / "frags"
    print(f"Reading dataset from {ds_root}...")
    ds = read_ds(ds_root)
    print(f"Number of items in the dataset: {len(ds)}")
    create_vanilla_index(ds)
    create_prefixed_index(ds)


if __name__ == "__main__":
    main()
