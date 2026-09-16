import argparse
import collections
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset

from src.common.config import load_yaml

# SoNaR text-category codes -> human-readable names (SoNaR User Documentation v1.0.4,
# Table 3.1 "Overview of file names, file sizes and numbers" / Table 2.2).
CATEGORY_NAMES = {
    "WR-P-E-A": "discussion_lists",
    "WR-P-E-C": "e_magazines",
    "WR-P-E-E": "newsletters_electronic",
    "WR-P-E-F": "press_releases",
    "WR-P-E-G": "subtitles",
    "WR-P-E-H": "teletext_pages",
    "WR-P-E-I": "web_sites",
    "WR-P-E-J": "wikipedia",
    "WR-P-E-K": "blogs",
    "WR-P-E-L": "tweets",
    "WR-P-P-B": "books",
    "WR-P-P-C": "brochures",
    "WR-P-P-D": "newsletters_printed",
    "WR-P-P-E": "guides_manuals",
    "WR-P-P-F": "legal_texts",
    "WR-P-P-G": "newspapers",
    "WR-P-P-H": "periodicals_magazines",
    "WR-P-P-I": "policy_documents",
    "WR-P-P-J": "proceedings",
    "WR-P-P-K": "reports",
    "WR-U-E-A": "chats",
    "WR-U-E-D": "sms",
    "WR-U-E-E": "written_assignments",
    "WS-U-E-A": "autocues",
    "WS-U-T-B": "texts_for_visually_impaired",
}

# Matches SONAR500/FoLiA/<CODE>_<human readable name>/<3-digit bucket>/<CODE>-<10 digit
# id>.folia.xml, wherever it sits in the archive (search, not match, so an extra wrapper
# directory doesn't break it). The category directory itself isn't captured - the code is
# read from the filename instead, where it's guaranteed present.
FOLIA_MEMBER_RE = re.compile(
    r"SONAR500/FoLiA/(?:[A-Z]{2}-[A-Z]-[A-Z]-[A-Z]_[a-z_]+)/\d{3}/"
    r"([A-Z]{2}-[A-Z]-[A-Z]-[A-Z])-(\d{10})\.folia\.xml$"
)

_NO_SPACE_BEFORE = set(".,;:!?)]}’”")
_NO_SPACE_AFTER = set("([{‘“")


def local_tag(elem: ET.Element) -> str:
    """Strip the FoLiA XML namespace, e.g. '{http://ilk.uvt.nl/folia}t' -> 't'."""
    tag = elem.tag
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _detokenize(words: list[str]) -> str:
    """Join word tokens into a sentence, without a space before closing punctuation
    or after an opening bracket/quote."""
    result = ""
    for i, word in enumerate(words):
        if i == 0:
            result = word
            continue
        prev_char = result[-1]
        if word in _NO_SPACE_BEFORE or prev_char in _NO_SPACE_AFTER:
            result += word
        else:
            result += " " + word
    return result


def _sentence_text(s: ET.Element) -> str:
    """Text of one <s> element: prefer a direct sentence-level <t> (exact original
    text) if present, else reconstruct by detokenizing its <w>/<t> word tokens."""
    for child in s:
        if local_tag(child) == "t" and (child.text or "").strip():
            return child.text.strip()

    words = []
    for w in s.iter():
        if local_tag(w) != "w":
            continue
        for child in w:
            if local_tag(child) == "t" and child.text:
                words.append(child.text)
                break
    return _detokenize(words)


def extract_text_from_folia(fileobj) -> str:
    """Reconstruct a FoLiA document's plain text: paragraphs (<p>) of sentences
    (<s>), joined with newlines between paragraphs and spaces between sentences.
    Non-paragraph-structured documents (e.g. chats, tweets) fall back to one
    "paragraph" of all sentences found anywhere in the document."""
    tree = ET.parse(fileobj)
    root = tree.getroot()

    text_root = next((e for e in root.iter() if local_tag(e) == "text"), root)

    paragraphs = []
    for p in text_root.iter():
        if local_tag(p) != "p":
            continue
        sentences = [_sentence_text(s) for s in p.iter() if local_tag(s) == "s"]
        sentences = [s for s in sentences if s]
        if sentences:
            paragraphs.append(" ".join(sentences))

    if not paragraphs:
        sentences = [_sentence_text(s) for s in text_root.iter() if local_tag(s) == "s"]
        paragraphs = [s for s in sentences if s]

    return "\n".join(paragraphs).strip()


def iter_folia_examples(archive_path: str):
    n_seen = 0
    n_written = 0
    n_errors = 0
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            match = FOLIA_MEMBER_RE.search(member.name)
            if not match:
                continue
            category, doc_id = match.groups()
            n_seen += 1

            fileobj = tf.extractfile(member)
            if fileobj is None:
                continue
            try:
                text = extract_text_from_folia(fileobj)
            except ET.ParseError as e:
                n_errors += 1
                print(f"  skipping {member.name}: XML parse error: {e}")
                continue
            finally:
                fileobj.close()

            if not text:
                continue
            n_written += 1
            if n_written % 5000 == 0:
                print(f"  ...{n_written:,} documents extracted ({n_seen:,} FoLiA files seen, {n_errors} errors)")

            yield {
                "text": text,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, category),
                "doc_id": doc_id,
                "source_file": member.name,
            }

    print(f"Done: {n_written:,} documents extracted from {n_seen:,} FoLiA files scanned ({n_errors} parse errors).")


def main():
    parser = argparse.ArgumentParser(
        description="Stream-extract raw text from every FoLiA document in a SoNaR-500 "
        "archive (SONAR500/DATA/**/*.folia.xml), without ever unpacking the archive to "
        "disk. Writes one HF dataset with columns: text, category, category_name, doc_id, "
        "source_file. Run this once; category selection and sampling for training happen "
        "later, cheaply, on this output (see the `local` source type in data_continued.yaml)."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/sonar_extract.yaml.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    archive_path = cfg["archive_path"]
    output_dir = cfg["output_dir"]

    print(f"Streaming FoLiA documents from {archive_path}...")
    dataset = Dataset.from_generator(iter_folia_examples, gen_kwargs={"archive_path": archive_path})

    print(f"Saving {len(dataset):,} documents to {output_dir}...")
    dataset.save_to_disk(output_dir)

    print("\nPer-category document counts:")
    counts = collections.Counter(dataset["category"])
    for category, count in sorted(counts.items()):
        print(f"  {category} ({CATEGORY_NAMES.get(category, '?')}): {count:,}")


if __name__ == "__main__":
    main()
