from pathlib import Path

ds_root = Path(__file__).parent / "ds" / "manuals"
mds_root = ds_root / "mds"


def get_good_files()->list[Path]:
    good_files = (ds_root / "good.txt").read_text(encoding="utf-8").splitlines()
    good_files = [f.strip() for f in good_files]
    good_files = [mds_root / f"{f}.md" for f in good_files if f and not f.startswith("#")]
    return good_files


def main():
    good_files = get_good_files()
    for md in good_files:
        print(md, md.stat().st_size)


if __name__ == "__main__":
    main()
