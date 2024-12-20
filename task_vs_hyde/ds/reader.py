from pathlib import Path

import yaml

from task_vs_hyde.ds import DatasetItem


def read_ds(root: Path) -> list[DatasetItem]:
    res = []
    for doc_yaml in root.glob("*.yaml"):
        frags = yaml.safe_load(doc_yaml.read_text(encoding="utf-8"))
        res += [DatasetItem.model_validate(frag) for frag in frags]
    return res
