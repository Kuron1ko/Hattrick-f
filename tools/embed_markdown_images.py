from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "results" / "geant" / "8sp" / "0" / "Hattrick复现与改进.md"
OUTPUT_MD = ROOT / "output" / "markdown" / "Hattrick复现与改进_内嵌图片.md"


IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def normalize_image_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    return target


def image_to_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    source = SOURCE_MD.read_text(encoding="utf-8")
    base_dir = SOURCE_MD.parent

    def replace_image(match: re.Match[str]) -> str:
        alt_text = match.group(1)
        raw_target = match.group(2)
        target = normalize_image_target(raw_target)
        image_path = Path(target)
        if not image_path.is_absolute():
            image_path = base_dir / image_path

        if not image_path.exists():
            return match.group(0)

        data_uri = image_to_data_uri(image_path)
        return f"![{alt_text}]({data_uri})"

    embedded = IMAGE_PATTERN.sub(replace_image, source)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(embedded, encoding="utf-8")
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()
