# this script prepares list of questions for selecting with prompt
from pathlib import Path
import csv
import random

from pydantic import BaseModel

from task_vs_hyde.ds.reader import read_ds

ds_root = Path(__file__).parent / "ds" / "manuals" / "frags"


class Question(BaseModel):
    doc_name: str
    frag_offset: int
    num: int
    question: str


def store_questions_csv(questions: list[Question], output_path: Path):
    qs = [(f"{q.doc_name}:{q.frag_offset}:{q.num}", q.question) for q in questions]
    with output_path.open("w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question"])
        writer.writerows(qs)


def select_questions(questions: list[Question], n: int, output_path: Path):
    if n > len(questions):
        raise ValueError(f"n={n} > len(questions)={len(questions)}")
    res = random.sample(questions, n)
    store_questions_csv(res, output_path)


def main():
    ds = read_ds(ds_root)
    print(f"Total fragments: {len(ds)}")
    questions = [
        Question(
            doc_name=frag.doc_name,
            frag_offset=frag.offset,
            num=idx,
            question=qa_pair.question,
        )
        for frag in ds
        for idx, qa_pair in enumerate(frag.qa_pairs)
    ]
    print(f"Total questions: {len(questions)}")
    csv_fname = ds_root / "questions.csv"
    store_questions_csv(questions, csv_fname)
    print(f"Questions stored in {csv_fname}")

    print("Selecting random subset of (200) questions...")
    select_questions(questions, 200, ds_root / "questions_selected.csv")


if __name__ == "__main__":
    main()
