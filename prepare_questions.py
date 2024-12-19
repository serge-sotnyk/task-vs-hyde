from pathlib import Path
import yaml
from yaml.representer import Representer

from task_vs_hyde.ds import DatasetItem
from task_vs_hyde.utils.splitters import ParagraphSplitter

ds_root = Path(__file__).parent / "ds" / "manuals"
mds_root = ds_root / "mds"


def get_good_files() -> list[Path]:
    good_files = (ds_root / "good.txt").read_text(encoding="utf-8").splitlines()
    good_files = [f.strip() for f in good_files]
    good_files = [mds_root / f"{f}.md" for f in good_files if f and not f.startswith("#")]
    return good_files


def prepare_fragments(md: Path) -> list[DatasetItem]:
    md_text = md.read_text(encoding="utf-8")
    res = []
    splitter = ParagraphSplitter(max_chars=4096)
    for frag in splitter.split(md_text):
        res.append(
            DatasetItem(
                doc_name=md.name,
                text=frag.text,
                offset=frag.offset,
            ))
    return res


def store_fragments_yaml(fragments: list[DatasetItem], output_path: Path):
    class LiteralString(str):
        pass

    def represent_literal_string(dumper: Representer, data: LiteralString):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

    yaml.add_representer(LiteralString, represent_literal_string)
    yaml.representer.SafeRepresenter.add_representer(LiteralString, represent_literal_string)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            [
                {
                    "doc_name": frag.doc_name,
                    "offset": frag.offset,
                    "qa_pairs": frag.qa_pairs,
                    "text": LiteralString(frag.text.replace("\r\n", "\n")),
                    # Normalize line endings
                }
                for frag in fragments
            ],
            f,
            indent=2,
            default_flow_style=False,
            allow_unicode=True,  # Preserve any non-ASCII characters
        )


def main():
    good_files = get_good_files()
    fragments: list[DatasetItem] = []
    for md in good_files:
        print(md, md.stat().st_size)
        doc_fragments = prepare_fragments(md)
        fragments += doc_fragments
        print(f"Doc split on {len(doc_fragments)} fragments.")

    output_path = ds_root / "manuals_ds.yaml"
    store_fragments_yaml(fragments, output_path)
    print(f"Fragments stored in {output_path}")


if __name__ == "__main__":
    main()
