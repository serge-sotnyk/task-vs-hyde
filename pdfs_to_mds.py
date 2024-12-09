# this converter uses docling library to convert pdf to mds
# https://ds4sd.github.io/docling/
from pathlib import Path

from docling.document_converter import DocumentConverter
from tqdm import tqdm

manuals_root = Path(__file__).parent / "ds" / "manuals"
pdfs_root = manuals_root / "pdfs"
mds_root = manuals_root / "mds"
converter = DocumentConverter()


def main():
    pdfs = sorted(pdfs_root.glob("*.pdf"), key=lambda x: x.stat().st_size)
    for pdf in tqdm(pdfs):
        md = mds_root / pdf.with_suffix(".md").name
        print(f"Converting {str(pdf)} to {str(md)}")
        result = converter.convert(pdf)
        md_txt = result.document.export_to_markdown()
        md.write_text(md_txt, encoding="utf-8")


if __name__ == "__main__":
    main()
