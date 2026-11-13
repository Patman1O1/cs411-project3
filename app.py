"""
UIC CS 411 - Project 03: Naive Bayes Text Classifier from Scratch

Multinomial Naive Bayes sentiment classifier (positive vs. negative) built
with the Python standard library only.

    P(c)     = (# training sentences in class c) / (# training sentences)
    P(w | c) = (count(w, c) + 1) / (N_c + |V|)          (Laplace, alpha = 1)
    score(c) = log P(c) + sum over tokens w in V of log P(w | c)

Run `python app.py` to train, evaluate, and enter the interactive loop.
Type `:help` in the loop to see the inspection commands (useful for the video).
"""

from __future__ import annotations

# Builtin Imports
import csv
import math
import re
import sys
from pathlib import Path
from typing import Final, TypedDict

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

POSITIVE: Final[str] = "positive"
NEGATIVE: Final[str] = "negative"
CLASSES: Final[tuple[str, str]] = (POSITIVE, NEGATIVE)
ALPHA: Final[int] = 1  # Laplace smoothing constant

# Directory containing app.py (used to resolve data/ paths from any cwd).
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent

DEFAULT_TRAIN_FILE: Final[str] = "data/train.csv"
DEFAULT_TEST_FILE: Final[str] = "data/test.csv"

# Expected dataset shape from the project guidelines (used only for warnings).
EXPECTED_COUNTS: Final[dict[str, dict[str, int]]] = {
    "train": {POSITIVE: 80, NEGATIVE: 80},
    "test": {POSITIVE: 20, NEGATIVE: 20},
}

# A token is a run of letters/digits, optionally joined by apostrophes so that
# contractions like "don't" or "wasn't" stay as one word (they carry negation).
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


# ----------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------

PathLike = str | Path
Example = tuple[str, str]  # (text, label)


class Model(TypedDict):
    train_file: str
    n_docs: int
    doc_counts: dict[str, int]
    word_counts: dict[str, dict[str, int]]
    total_words: dict[str, int]
    vocab: set[str]
    vocab_size: int
    priors: dict[str, float]
    log_priors: dict[str, float]
    log_likelihoods: dict[str, dict[str, float]]


class ScoreResult(TypedDict):
    tokens: list[str]
    known: list[str]
    unknown: list[str]
    scores: dict[str, float]


class ConfusionMatrix(TypedDict):
    TP: int
    TN: int
    FP: int
    FN: int


class Misclassified(TypedDict):
    text: str
    label: str
    predicted: str


class EvaluationResult(TypedDict):
    accuracy: float
    confusion_matrix: ConfusionMatrix
    total: int
    correct: int
    precision: float
    recall: float
    f1: float
    misclassified: list[Misclassified]


# Global model state, filled in by train_model().
MODEL: Model | None = None


# ----------------------------------------------------------------------------
# Data loading and preprocessing
# ----------------------------------------------------------------------------

def tokenize(text: str | None) -> list[str]:
    """
    Lowercase the text and split it into word tokens.

    Steps:
      1. Lowercase.
      2. Normalize curly apostrophes (’ ‘) to a straight apostrophe.
      3. Keep runs of letters/digits (with inner apostrophes); punctuation and
         whitespace act as separators.

    Repeated words are kept: "good good movie" -> ["good", "good", "movie"].
    """
    if text is None:
        return []
    text = str(text).lower().replace("’", "'").replace("‘", "'")
    return TOKEN_PATTERN.findall(text)


def _resolve_path(path: PathLike) -> Path:
    """Use the path as given if it exists; otherwise try it relative to app.py."""
    path = Path(path)
    if path.exists():
        return path

    alt: Path = SCRIPT_DIR / path
    if alt.exists():
        return alt
    return path  # let open() raise a clear FileNotFoundError


def load_dataset(path: PathLike) -> list[Example]:
    """
    Read a `text,label` CSV and return a list of (text, label) pairs.

    - Handles a UTF-8 BOM and an optional header row.
    - Labels are lowercased/stripped; rows with any other label are skipped
      with a warning, as are blank rows.
    """
    resolved: Path = _resolve_path(path)
    rows: list[Example] = []

    with resolved.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 2:
                print(f"[warning] {resolved}:{line_no}: expected 2 columns, got {len(row)}; skipped",
                      file=sys.stderr)
                continue
            # If a sentence contains an unquoted comma, the label is still the last cell.
            text: str = ",".join(row[:-1]).strip()
            label: str = row[-1].strip().lower()
            if line_no == 1 and text.lower() == "text" and label == "label":
                continue  # header
            if label not in CLASSES:
                print(f"[warning] {resolved}:{line_no}: unknown label {label!r}; skipped",
                      file=sys.stderr)
                continue
            rows.append((text, label))
    return rows


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_model(train_file: PathLike = DEFAULT_TRAIN_FILE) -> Model:
    """
    Train the Naive Bayes classifier on the training dataset.

    Computes, from the training file only:
      - vocabulary V (all unique training tokens)
      - count(w, c) for every word and class (every occurrence counted)
      - N_c = total number of word occurrences in class c
      - class priors P(c)
      - Laplace-smoothed log-likelihoods log P(w | c)

    Returns the model dictionary (also stored in the global MODEL).
    """
    global MODEL

    data: list[Example] = load_dataset(train_file)

    doc_counts: dict[str, int] = {c: 0 for c in CLASSES}
    word_counts: dict[str, dict[str, int]] = {c: {} for c in CLASSES}
    total_words: dict[str, int] = {c: 0 for c in CLASSES}
    vocab: set[str] = set()

    for text, label in data:
        doc_counts[label] += 1
        counts: dict[str, int] = word_counts[label]
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
            total_words[label] += 1
            vocab.add(token)

    n_docs: int = sum(doc_counts.values())
    V: int = len(vocab)

    # Class priors. A class with no training sentences gets prior 0 -> log = -inf.
    priors: dict[str, float] = {c: (doc_counts[c] / n_docs if n_docs else 0.0) for c in CLASSES}
    log_priors: dict[str, float] = {
        c: (math.log(priors[c]) if priors[c] > 0 else float("-inf")) for c in CLASSES
    }

    # Smoothed likelihoods for every word in V, for each class.
    log_likelihoods: dict[str, dict[str, float]] = {c: {} for c in CLASSES}
    for c in CLASSES:
        denom: int = total_words[c] + ALPHA * V
        for w in vocab:
            log_likelihoods[c][w] = math.log((word_counts[c].get(w, 0) + ALPHA) / denom)

    model: Model = {
        "train_file": str(train_file),
        "n_docs": n_docs,
        "doc_counts": doc_counts,
        "word_counts": word_counts,
        "total_words": total_words,
        "vocab": vocab,
        "vocab_size": V,
        "priors": priors,
        "log_priors": log_priors,
        "log_likelihoods": log_likelihoods,
    }
    MODEL = model
    return model


def _ensure_trained() -> Model:
    """Return the global model, training it on the default file if needed."""
    if MODEL is None:
        return train_model()
    return MODEL


def _get_model(model: Model | None) -> Model:
    """Use the given model, or fall back to the global (trained on demand)."""
    return model if model is not None else _ensure_trained()


def likelihood_parts(word: str, c: str, model: Model | None = None) -> tuple[int, int, float]:
    """Return (numerator, denominator, probability) for P(word | c)."""
    m: Model = _get_model(model)
    num: int = m["word_counts"][c].get(word, 0) + ALPHA
    den: int = m["total_words"][c] + ALPHA * m["vocab_size"]
    return num, den, num / den


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

def score_sentence(text: str, model: Model | None = None) -> ScoreResult:
    """
    Compute the log score of each class for a sentence.

    Returns a dict with the tokens, the known/unknown split, and per-class
    scores, so the computation can be traced step by step.
    """
    m: Model = _get_model(model)
    tokens: list[str] = tokenize(text)
    known: list[str] = [t for t in tokens if t in m["vocab"]]
    unknown: list[str] = [t for t in tokens if t not in m["vocab"]]

    scores: dict[str, float] = {}
    for c in CLASSES:
        s: float = m["log_priors"][c]
        ll: dict[str, float] = m["log_likelihoods"][c]
        for t in known:  # every occurrence counts, including repeats
            s += ll[t]
        scores[c] = s

    return {"tokens": tokens, "known": known, "unknown": unknown, "scores": scores}


def _is_tie(pos: float, neg: float) -> bool:
    """True if the two log scores are equal (within floating-point tolerance)."""
    return (pos == neg) or (
        math.isfinite(pos) and math.isfinite(neg)
        and math.isclose(pos, neg, rel_tol=1e-12, abs_tol=1e-12)
    )


def _decide(scores: dict[str, float], priors: dict[str, float]) -> str:
    """Pick the winning class, applying the tie-breaking rule."""
    pos: float = scores[POSITIVE]
    neg: float = scores[NEGATIVE]
    if not _is_tie(pos, neg):
        return POSITIVE if pos > neg else NEGATIVE
    # Tie: higher prior wins; equal priors default to positive.
    if priors[NEGATIVE] > priors[POSITIVE]:
        return NEGATIVE
    return POSITIVE


def predict(text: str) -> str:
    """
    Predict sentiment class for an input sentence.
    Returns: 'positive' or 'negative'
    Tie-breaking: If scores are tied, select the class with the higher prior P(c),
                  or default to 'positive' if priors are equal.
    Words not in the training vocabulary are ignored.
    """
    model: Model = _ensure_trained()
    result: ScoreResult = score_sentence(text, model)
    return _decide(result["scores"], model["priors"])


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def evaluate_model(test_file: PathLike = DEFAULT_TEST_FILE) -> EvaluationResult:
    """
    Evaluate classifier performance on a test dataset.
    'positive' is treated as the positive class for the confusion matrix.

    Returns:
        dict: {
            "accuracy": float,
            "confusion_matrix": {"TP": int, "TN": int, "FP": int, "FN": int},
            plus extra fields: total, correct, precision, recall, f1,
            misclassified (list of {text, label, predicted})
        }
    """
    _ensure_trained()
    data: list[Example] = load_dataset(test_file)

    cm: ConfusionMatrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    misclassified: list[Misclassified] = []
    for text, label in data:
        pred: str = predict(text)
        if label == POSITIVE and pred == POSITIVE:
            cm["TP"] += 1
        elif label == NEGATIVE and pred == NEGATIVE:
            cm["TN"] += 1
        elif label == NEGATIVE and pred == POSITIVE:
            cm["FP"] += 1
        else:
            cm["FN"] += 1
        if pred != label:
            misclassified.append({"text": text, "label": label, "predicted": pred})

    total: int = len(data)
    correct: int = cm["TP"] + cm["TN"]
    accuracy: float = correct / total if total else 0.0
    precision: float = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) else 0.0
    recall: float = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) else 0.0
    f1: float = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "total": total,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "misclassified": misclassified,
    }


# ----------------------------------------------------------------------------
# Reporting helpers (for running app.py and for the video walkthrough)
# ----------------------------------------------------------------------------

def _fmt_log(x: float) -> str:
    return "-inf" if x == float("-inf") else f"{x:.4f}"


def check_dataset(train_file: PathLike = DEFAULT_TRAIN_FILE,
                  test_file: PathLike = DEFAULT_TEST_FILE) -> None:
    """Print class balance for both files and warn about guideline mismatches."""
    print("=" * 70)
    print("DATASET")
    print("=" * 70)
    files: dict[str, str] = {"train": str(train_file), "test": str(test_file)}
    loaded: dict[str, list[Example]] = {}
    for split, path in files.items():
        try:
            rows: list[Example] = load_dataset(path)
        except FileNotFoundError:
            print(f"  {split:5s}: {path} NOT FOUND")
            continue
        loaded[split] = rows
        counts: dict[str, int] = {c: sum(1 for _, l in rows if l == c) for c in CLASSES}
        print(f"  {split:5s}: {path:16s} {len(rows):4d} sentences  "
              f"(positive={counts[POSITIVE]}, negative={counts[NEGATIVE]})")
        exp: dict[str, int] = EXPECTED_COUNTS[split]
        if counts != exp:
            print(f"         [warning] guidelines expect positive={exp[POSITIVE]}, "
                  f"negative={exp[NEGATIVE]}")

    if len(loaded) == 2:
        total_pos: int = sum(1 for r in loaded.values() for _, l in r if l == POSITIVE)
        total_neg: int = sum(1 for r in loaded.values() for _, l in r if l == NEGATIVE)
        print(f"  total: {total_pos + total_neg} sentences "
              f"(positive={total_pos}, negative={total_neg})")
        train_texts: set[str] = {t.strip().lower() for t, _ in loaded["train"]}
        overlap: list[str] = [t for t, _ in loaded["test"] if t.strip().lower() in train_texts]
        if overlap:
            print(f"  [warning] {len(overlap)} test sentence(s) also appear in train.csv "
                  f"(train/test must be separate):")
            for t in overlap:
                print(f"            - {t}")
        dup_train: int = len(loaded["train"]) - len(train_texts)
        if dup_train:
            print(f"  [note] train.csv contains {dup_train} duplicate sentence(s)")
    print()


def print_model_summary(model: Model | None = None) -> None:
    m: Model = _get_model(model)
    print("=" * 70)
    print("MODEL (computed from training data only)")
    print("=" * 70)
    V: int = m["vocab_size"]
    print(f"  Training sentences : {m['n_docs']}")
    print(f"  Vocabulary size |V|: {V}")
    for c in CLASSES:
        print(f"  N_{c[:3]} (total {c} word occurrences): {m['total_words'][c]}")
    print()
    print("  Class priors  P(c) = (# sentences in c) / (# training sentences)")
    for c in CLASSES:
        print(f"    P({c}) = {m['doc_counts'][c]}/{m['n_docs']} = "
              f"{m['priors'][c]:.4f}   log = {_fmt_log(m['log_priors'][c])}")
    print()
    print("  Smoothed denominators  N_c + |V|")
    for c in CLASSES:
        print(f"    {c:8s}: {m['total_words'][c]} + {V} = {m['total_words'][c] + V}")
    print()


def print_word_details(words: list[str], model: Model | None = None) -> None:
    """Show count(w, c) and P(w | c) = (count + 1) / (N_c + |V|) for each word."""
    m: Model = _get_model(model)
    for raw in words:
        for w in tokenize(raw) or [raw.lower()]:
            in_v: bool = w in m["vocab"]
            print(f"  word: {w!r}  {'(in V)' if in_v else '(NOT in V - ignored at prediction)'}")
            for c in CLASSES:
                cnt: int = m["word_counts"][c].get(w, 0)
                num, den, p = likelihood_parts(w, c, m)
                print(f"    {c:8s}: count = {cnt:3d}   P(w|{c[:3]}) = ({cnt} + 1) / "
                      f"({m['total_words'][c]} + {m['vocab_size']}) = {num}/{den} = {p:.6f}"
                      f"   log = {math.log(p):.4f}")


def print_trace(text: str, model: Model | None = None) -> str:
    """Step-by-step log-score computation for one sentence."""
    m: Model = _get_model(model)
    result: ScoreResult = score_sentence(text, m)
    print(f"  tokens  : {result['tokens']}")
    print(f"  in V    : {result['known']}")
    print(f"  ignored : {result['unknown'] if result['unknown'] else '(none)'}")
    print()
    header: str = f"    {'token':<16}" + "".join(f"{'P(w|' + c[:3] + ')':>22}" for c in CLASSES)
    print(header)
    for t in result["known"]:
        cells: str = ""
        for c in CLASSES:
            num, den, p = likelihood_parts(t, c, m)
            cells += f"{f'{num}/{den}':>12}{math.log(p):>10.4f}"
        print(f"    {t:<16}{cells}")
    print()
    for c in CLASSES:
        terms: str = " + ".join([f"log P({c[:3]})"] + [f"log P({t}|{c[:3]})" for t in result["known"]])
        print(f"  score({c}) = {terms}")
        print(f"             = {_fmt_log(m['log_priors'][c])} + "
              f"{sum(m['log_likelihoods'][c][t] for t in result['known']):.4f}"
              f" = {_fmt_log(result['scores'][c])}")
    decision: str = _decide(result["scores"], m["priors"])
    tied: bool = _is_tie(result["scores"][POSITIVE], result["scores"][NEGATIVE])
    note: str = " (scores tied -> higher prior wins, else positive)" if tied else ""
    print(f"  prediction: {decision.upper()}{note}")
    return decision


def print_top_words(n: int = 10, model: Model | None = None) -> None:
    """Words with the largest log-likelihood ratio toward each class."""
    m: Model = _get_model(model)
    ratios: list[tuple[float, str]] = []
    for w in m["vocab"]:
        r: float = m["log_likelihoods"][POSITIVE][w] - m["log_likelihoods"][NEGATIVE][w]
        ratios.append((r, w))
    ratios.sort()
    print("  Most positive words (log P(w|pos) - log P(w|neg)):")
    for r, w in reversed(ratios[-n:]):
        print(f"    {w:<18} {r:+.4f}  (pos={m['word_counts'][POSITIVE].get(w, 0)}, "
              f"neg={m['word_counts'][NEGATIVE].get(w, 0)})")
    print("  Most negative words:")
    for r, w in ratios[:n]:
        print(f"    {w:<18} {r:+.4f}  (pos={m['word_counts'][POSITIVE].get(w, 0)}, "
              f"neg={m['word_counts'][NEGATIVE].get(w, 0)})")


def print_evaluation(results: EvaluationResult) -> None:
    print("=" * 70)
    print("EVALUATION on held-out test set")
    print("=" * 70)
    cm: ConfusionMatrix = results["confusion_matrix"]
    print(f"  Accuracy : {results['accuracy']:.4f}  ({results['correct']}/{results['total']})")
    print(f"  Precision: {results['precision']:.4f}   Recall: {results['recall']:.4f}   "
          f"F1: {results['f1']:.4f}   (positive = positive class)")
    print()
    print("  Confusion matrix:")
    print("                       predicted positive   predicted negative")
    print(f"    actual positive   {('TP = ' + str(cm['TP'])):>18}   {('FN = ' + str(cm['FN'])):>18}")
    print(f"    actual negative   {('FP = ' + str(cm['FP'])):>18}   {('TN = ' + str(cm['TN'])):>18}")
    if results["misclassified"]:
        print()
        print("  Misclassified test sentences:")
        for mis in results["misclassified"]:
            print(f"    [{mis['label']} -> {mis['predicted']}] {mis['text']}")
    print()


HELP_TEXT: Final[str] = """\
  Type a sentence to see its prediction and full log-score trace.
  Commands:
    :stats             vocabulary size, N_c, priors, denominators
    :word w1 w2 ...    count(w, c) and P(w | c) for specific words
    :top [n]           most class-indicative words (default 10)
    :eval              re-run evaluation on data/test.csv
    :retrain           reload data/train.csv (after editing it) and re-evaluate
    :dataset           class balance / overlap check for train and test files
    :help              show this help
    quit / exit        leave
"""


def interactive_loop(train_file: PathLike = DEFAULT_TRAIN_FILE,
                     test_file: PathLike = DEFAULT_TEST_FILE) -> None:
    print("=" * 70)
    print("INTERACTIVE MODE  (type :help for commands, 'quit' to exit)")
    print("=" * 70)
    while True:
        try:
            line: str = input("\nsentence> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        lower: str = line.lower()
        if lower in ("quit", "exit", ":q", ":quit", ":exit"):
            break
        if lower in (":help", "help", "?"):
            print(HELP_TEXT)
        elif lower == ":stats":
            print_model_summary()
        elif lower.startswith(":word"):
            words: list[str] = line.split()[1:]
            if words:
                print_word_details(words)
            else:
                print("  usage: :word good bad")
        elif lower.startswith(":top"):
            parts: list[str] = line.split()
            n: int = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            print_top_words(n)
        elif lower == ":eval":
            try:
                print_evaluation(evaluate_model(test_file))
            except FileNotFoundError:
                print(f"  test file {str(test_file)!r} not found")
        elif lower == ":retrain":
            try:
                train_model(train_file)
            except FileNotFoundError:
                print(f"  training file {str(train_file)!r} not found; keeping previous model")
                continue
            print_model_summary()
            try:
                print_evaluation(evaluate_model(test_file))
            except FileNotFoundError:
                print(f"  test file {str(test_file)!r} not found; skipping evaluation")
        elif lower == ":dataset":
            check_dataset(train_file, test_file)
        elif lower.startswith(":"):
            print("  unknown command; type :help")
        else:
            print_trace(line)


def main(argv: list[str] | None = None) -> int:
    args: list[str] = sys.argv[1:] if argv is None else argv
    train_file: str = DEFAULT_TRAIN_FILE
    test_file: str = DEFAULT_TEST_FILE

    check_dataset(train_file, test_file)
    try:
        train_model(train_file)
    except FileNotFoundError:
        print(f"Training file {train_file!r} not found. Create data/train.csv (text,label) first.")
        return 1
    print_model_summary()
    try:
        print_evaluation(evaluate_model(test_file))
    except FileNotFoundError:
        print(f"Test file {test_file!r} not found; skipping evaluation.\n")

    # Interactive loop; ends on 'quit' or end of input (Ctrl-D / closed stdin).
    # Pass --no-interactive to skip it.
    if "--no-interactive" not in args:
        interactive_loop(train_file, test_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
