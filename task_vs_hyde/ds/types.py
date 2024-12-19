from pydantic import BaseModel, Field

from task_vs_hyde.utils.splitters import TextChunk


class QAPair(BaseModel):
    question: str
    answer: str


class DatasetItem(BaseModel):
    doc_name: str
    text: str = ""
    offset: int = 0
    qa_pairs: list[QAPair] = Field(default_factory=list)
