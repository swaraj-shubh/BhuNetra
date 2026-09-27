"""Natural-language query -> {semantic, change, spatial, temporal, sensor} (DESIGN.md §3 query path).

Offline, rule-based and deterministic: vocabulary lives in configs/query.yaml, distances in thresholds.yaml.
Order matters: spatial phrases are cut first, so "near a road" never counts as a road *change*.
"""

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

from configs.settings import T, aois, lexicon


@dataclass(frozen=True)
class Parsed:
    text: str
    semantic: str  # what CLIP encodes
    change: bool  # any change cue present
    change_types: tuple[str, ...]  # empty with change=True -> any type
    near: dict[str, float]  # spatial target -> metres
    date_range: tuple[str | None, str | None] | None
    aoi: str | None
    sensor: str | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _alt(phrases: list[str]) -> str:
    """Regex alternation, longest first so 'forest loss' wins over 'forest'."""
    return "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))


def _has(text: str, phrases: list[str]) -> bool:
    return bool(re.search(rf"\b(?:{_alt(phrases)})\b", text))


def _cut(text: str, span: tuple[int, int]) -> str:
    return text[: span[0]] + " " + text[span[1] :]


def _spatial(s: str) -> tuple[str, dict[str, float]]:
    target_of = {w: t for t, ws in lexicon["targets"].items() for w in ws}
    rx = re.compile(
        rf"\b(?:within\s+(?P<d>\d+(?:\.\d+)?)\s*(?P<u>km|m)\s+of|(?:{_alt(lexicon['relations'])}))\s+"
        rf"(?:a|an|the)?\s*(?P<t>{_alt(list(target_of))})\b"
    )
    near: dict[str, float] = {}
    while m := rx.search(s):
        d = float(m["d"]) * (1000 if m["u"] == "km" else 1) if m["d"] else T.context.NEAR_M
        near[target_of[m["t"]]] = d
        s = _cut(s, m.span())
    return s, near


def _temporal(s: str) -> tuple[str, tuple[str | None, str | None] | None]:
    y = r"((?:19|20)\d{2})"
    rules: list[tuple[str, Callable[..., tuple[str | None, str | None]]]] = [
        (rf"\b(?:between|from)\s+{y}\s+(?:and|to|-)\s+{y}\b", lambda a, b: (f"{a}-01-01", f"{b}-12-31")),
        (rf"\bsince\s+{y}\b", lambda a: (f"{a}-01-01", None)),
        (rf"\bafter\s+{y}\b", lambda a: (f"{int(a) + 1}-01-01", None)),
        (rf"\bbefore\s+{y}\b", lambda a: (None, f"{int(a) - 1}-12-31")),
        (rf"\buntil\s+{y}\b", lambda a: (None, f"{a}-12-31")),
        (rf"\b(?:in|during)?\s*{y}\b", lambda a: (f"{a}-01-01", f"{a}-12-31")),
    ]
    for pattern, make in rules:
        if m := re.search(pattern, s):
            return _cut(s, m.span()), make(*m.groups())
    return s, None


def parse(text: str) -> Parsed:
    s = " ".join(text.lower().split())
    s, near = _spatial(s)
    s, dates = _temporal(s)

    sensor = None
    for name, words in lexicon["sensors"].items():
        if m := re.search(rf"\b(?:on|from|using|in)?\s*(?:{_alt(words)})\b", s):
            sensor, s = name, _cut(s, m.span())
            break
    aoi = None
    if aois and (m := re.search(rf"\b(?:in|at|over)?\s*({_alt(list(aois))})\b", s)):
        aoi, s = m.group(1), _cut(s, m.span())

    typed = tuple(t for t, ws in lexicon["change_types"].items() if _has(s, ws))
    # "construction since 2023": a type word plus a time phrase asks what changed, even without a cue word
    change = _has(s, lexicon["change_cues"]) or (dates is not None and bool(typed))
    types = typed if change else ()
    warnings = ("below_resolution",) if _has(s, lexicon["below_resolution"]) else ()

    sem = re.sub(rf"\b(?:{_alt(lexicon['semantic_drop'])})\b", " ", s)
    sem = " ".join(w for w in re.findall(r"[\w-]+", sem) if w not in set(lexicon["stopwords"]))
    if not sem:  # e.g. "what changed near the river": fall back to the type names, then the raw text
        sem = " ".join(types) or " ".join(text.split())
    return Parsed(text, sem, change, types, near, dates, aoi, sensor, warnings)
