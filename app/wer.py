"""Word error rate: how far a transcript is from the correct (reference) one.

Both texts are normalised first (case, punctuation, bracketed notes), so a WER counts wrong,
missing and extra words, not formatting. WER = (substitutions + deletions + insertions) / reference words.
"""

import re
import unicodedata
from dataclasses import dataclass, field

_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>")  # [music], (laughs), <unk>
_APOSTROPHES = re.compile(r"[’‘`´]")


def normalize(text: str) -> list[str]:
    """Lower-case words without punctuation; apostrophes inside words are kept (l'acqua, don't)."""
    text = unicodedata.normalize("NFKC", _APOSTROPHES.sub("'", text)).lower()
    text = _BRACKETS.sub(" ", text)
    chars = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch == "'" or category[0] in "LN" or category == "Mn":
            chars.append(ch)
        else:
            chars.append(" ")
    return [w.strip("'") for w in "".join(chars).split() if w.strip("'")]


@dataclass
class WerResult:
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int
    # Word-level alignment: [op, reference word, hypothesis word]; op is ok, sub, del or ins.
    alignment: list[list[str]] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_words


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    ref, hyp = normalize(reference), normalize(hypothesis)
    # Levenshtein distance over words, keeping the table for the alignment.
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        above, row = cost[i - 1], cost[i]
        for j in range(1, cols):
            row[j] = min(above[j - 1] + (ref[i - 1] != hyp[j - 1]), above[j] + 1, row[j - 1] + 1)

    # Walk back through the table. Among equally cheap steps prefer a match, then a deletion or insertion
    # over a substitution, so equal words stay aligned (easier to read in a diff); the error count is the same.
    alignment: list[list[str]] = []
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            alignment.append(["ok", ref[i - 1], hyp[j - 1]])
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            alignment.append(["del", ref[i - 1], ""])
            i -= 1
        elif j and cost[i][j] == cost[i][j - 1] + 1:
            alignment.append(["ins", "", hyp[j - 1]])
            j -= 1
        else:
            alignment.append(["sub", ref[i - 1], hyp[j - 1]])
            i, j = i - 1, j - 1
    alignment.reverse()
    count = {op: sum(a[0] == op for a in alignment) for op in ("sub", "del", "ins")}
    return WerResult(len(ref), count["sub"], count["del"], count["ins"], alignment)
