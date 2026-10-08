import argparse
import collections
import gzip
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset, load_from_disk

from src.common.config import load_yaml

# CGN component codes -> human-readable names (components.txt in the archive).
COMPONENT_NAMES = {
    "comp-a": "spontaneous_conversations",
    "comp-b": "interviews_teachers_of_dutch",
    "comp-c": "spontaneous_telephone_dialogues_switchboard",
    "comp-d": "spontaneous_telephone_dialogues_md",
    "comp-e": "simulated_business_negotiations",
    "comp-f": "interviews_discussions_debates_broadcast",
    "comp-g": "political_discussions_debates_meetings",
    "comp-h": "classroom_lessons",
    "comp-i": "live_commentaries_broadcast",
    "comp-j": "newsreports_reportages_broadcast",
    "comp-k": "news_broadcast",
    "comp-l": "commentaries_columns_reviews_broadcast",
    "comp-m": "ceremonious_speeches_sermons",
    "comp-n": "lectures_seminars",
    "comp-o": "read_speech",
}

# Matches .../data/annot/text/ort/<component>/<language>/<doc_id>.ort.gz, restricted to
# Netherlands Dutch (nl) - Flemish (vl) transcripts are skipped.
ORT_MEMBER_RE = re.compile(r"data/annot/text/ort/(comp-[a-o])/nl/([^/]+)\.ort\.gz$")

# A word may carry one of these suffixes (per the .ort format doc): *v foreign word, *d
# dialectal, *a incomplete/truncated fragment, *u slip-of-tongue/onomatopoeia, *z dialectal
# pronunciation, *x hard to understand. Only *a marks a non-word (a word fragment) - the
# rest mark a real, complete word, so for those only the marker itself is stripped.
_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\*(?P<marker>[aduvxzADUVXZ])(?P<punct>[.,?]*)$")
# ggg = non-speech sound, xxx/Xxx = one or more unintelligible words - not real words, even
# as a hyphen-joined sub-part of a word blend (e.g. "achten-xxx-tig").
_NOISE_TOKEN_RE = re.compile(r"^(ggg|xxx|Xxx)$")


def _clean_word(token: str) -> str | None:
    """Clean one whitespace-delimited token from a CGN .ort transcript. Returns None if the
    token is a non-word (unintelligible/truncated) and should be dropped - trailing sentence
    punctuation is preserved even then, so sentence boundaries survive."""
    if any(_NOISE_TOKEN_RE.match(part) for part in token.split("-")):
        punct = re.search(r"[.,?]+$", token)
        return punct.group(0) if punct else None

    match = _SUFFIX_RE.match(token)
    if not match:
        return token
    if match["marker"].lower() == "a":
        return match["punct"] or None
    return match["base"] + match["punct"]


def parse_short_textgrid(content: str) -> list[tuple[float, str]]:
    """Parse a Praat ShortTextGrid (.ort) file into (start_time, text) intervals across all
    tiers (speakers), unsorted - caller merges/sorts by time to reconstruct dialogue order."""
    lines = [line.strip() for line in content.splitlines()]
    pos = 0

    def next_line() -> str:
        nonlocal pos
        line = lines[pos]
        pos += 1
        return line

    def next_quoted() -> str:
        line = next_line()
        return line[1:-1] if line.startswith('"') and line.endswith('"') else line

    for _ in range(6):  # file type, "TextGrid", blank, global start/end, <exists>
        next_line()
    n_tiers = int(next_line())

    intervals = []
    for _ in range(n_tiers):
        next_line()  # "IntervalTier"
        next_quoted()  # speaker name
        next_line()  # tier start
        next_line()  # tier end
        n_intervals = int(next_line())
        for _ in range(n_intervals):
            start = float(next_line())
            next_line()  # interval end (unused - sorting only needs start)
            text = next_quoted()
            intervals.append((start, text))
    return intervals


def ort_to_text(content: str) -> str:
    """Reconstruct one document's plain text: all tiers' intervals merged in chronological
    order (speakers can overlap/cross-talk - that's expected), silences dropped, disfluency/
    annotation codes cleaned per `_clean_word`."""
    intervals = sorted(parse_short_textgrid(content), key=lambda iv: iv[0])
    words = []
    for _, text in intervals:
        if not text:
            continue
        for token in text.split():
            cleaned = _clean_word(token)
            if cleaned:
                words.append(cleaned)
    return " ".join(words)


def iter_cgn_examples(archive_path: str):
    n_seen = 0
    n_written = 0
    n_errors = 0
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            match = ORT_MEMBER_RE.search(name)
            if not match:
                continue
            component, doc_id = match.groups()
            n_seen += 1

            try:
                content = gzip.decompress(zf.read(name)).decode("latin-1")
                text = ort_to_text(content)
            except Exception as e:
                n_errors += 1
                print(f"  skipping {name}: {e}")
                continue

            if not text:
                continue
            n_written += 1
            if n_written % 2000 == 0:
                print(f"  ...{n_written:,} documents extracted ({n_seen:,} .ort files seen, {n_errors} errors)")

            yield {
                "text": text,
                "category": component,
                "category_name": COMPONENT_NAMES.get(component, component),
                "doc_id": doc_id,
            }

    print(f"Done: {n_written:,} documents extracted from {n_seen:,} .ort files scanned ({n_errors} errors).")


def main():
    parser = argparse.ArgumentParser(
        description="Stream-extract Dutch (nl) orthographic transcripts from every .ort.gz "
        "file in a CGN (Corpus Gesproken Nederlands) .zip archive, without ever unpacking "
        "the archive to disk (zip's central-directory index allows direct per-member reads). "
        "Disfluency/annotation codes are cleaned (suffix markers stripped, unintelligible/"
        "incomplete-word placeholders dropped) while reconstructing each document's plain "
        "text. Writes one HF dataset with columns: text, category, category_name, doc_id. "
        "Run this once; category selection and sampling for training happen later, cheaply, "
        "via the `local` source type in data_base.yaml/data_continued.yaml."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to configs/cgn_extract.yaml.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    archive_path = cfg["archive_path"]
    output_dir = cfg["output_dir"]

    if Path(output_dir).exists():
        try:
            dataset = load_from_disk(output_dir)
            print(f"Found existing extracted dataset at {output_dir} ({len(dataset):,} documents), skipping.")
            return
        except Exception as e:
            print(f"Found {output_dir} but couldn't load it ({e}) - re-running.")

    print(f"Streaming .ort transcripts from {archive_path}...")
    dataset = Dataset.from_generator(iter_cgn_examples, gen_kwargs={"archive_path": archive_path})

    print(f"Saving {len(dataset):,} documents to {output_dir}...")
    dataset.save_to_disk(output_dir)

    print("\nPer-component document counts:")
    counts = collections.Counter(dataset["category"])
    for category, count in sorted(counts.items()):
        print(f"  {category} ({COMPONENT_NAMES.get(category, '?')}): {count:,}")


if __name__ == "__main__":
    main()
