"""Per-source adapters.

Each source knows only how to yield ``(prompt, code)`` pairs plus its licence.
Everything else — dialect detection, animation counting, dedupe keys — is the
schema's job, so adding a source later means writing one small function.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from .schema import CorpusRow

PARQUET = ("https://huggingface.co/datasets/{repo}/resolve/"
           "refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet")


def _url(repo: str, split: str = "train", config: str = "default") -> str:
    return PARQUET.format(repo=repo, split=split, config=config)


# --- individual sources ------------------------------------------------------

def manimbench(split: str = "train") -> Iterator[CorpusRow]:
    """ManimBench v1 — hand-curated, the benchmark everyone measures against.

    Uses ``Reviewed Description``, not ``Generated Description``: the latter is
    populated in only 19 of 316 training rows despite being the column the
    dataset card points you at.
    """
    df = pd.read_parquet(_url("SuienR/ManimBench-v1", split))
    for i, r in df.iterrows():
        desc = r.get("Reviewed Description") or r.get("Generated Description")
        if not desc or not r.get("Code"):
            continue
        yield CorpusRow.build(
            source=f"manimbench-{split}", license="CC-BY-NC-SA-4.0",
            prompt=str(desc), code=str(r["Code"]), index=int(i),
            tags=["curated", f"difficulty:{str(r.get('Type','?')).lower()}"],
        )


def thanhkt() -> Iterator[CorpusRow]:
    """Largest single set (4.4k) and the lowest provenance — bare input/output."""
    df = pd.read_parquet(_url("thanhkt/manim_code"))
    for i, r in df.iterrows():
        if not r.get("input") or not r.get("output"):
            continue
        yield CorpusRow.build(source="thanhkt", license="unstated",
                              prompt=str(r["input"]), code=str(r["output"]),
                              index=int(i), tags=["bulk"])


def generaleoley() -> Iterator[CorpusRow]:
    """2024 vintage — expect pre-0.19 API drift, which the render gate will find."""
    df = pd.read_parquet(_url("generaleoley/manim-codegen"))
    for i, r in df.iterrows():
        if not r.get("query") or not r.get("answer"):
            continue
        yield CorpusRow.build(source="generaleoley", license="unstated",
                              prompt=str(r["query"]), code=str(r["answer"]),
                              index=int(i), tags=["bulk"])


def bespoke() -> Iterator[CorpusRow]:
    """Synthetic, but uniquely ships narration and the original stderr.

    Only the text columns are read — the full file is 181 MB because it embeds
    rendered video, and we re-render everything through our own harness anyway.
    """
    df = pd.read_parquet(
        _url("bespokelabs/bespoke-manim"),
        columns=["question", "title", "narration", "python_code", "error"],
    )
    for i, r in df.iterrows():
        if not r.get("python_code"):
            continue
        # The narration is what the animation must actually depict, which makes
        # a far better prompt than the bare question.
        prompt = " ".join(str(x) for x in (r.get("title"), r.get("question")) if x)
        tags = ["synthetic"]
        if r.get("error"):
            tags.append("known-broken-upstream")
        yield CorpusRow.build(source="bespoke", license="unstated",
                              prompt=prompt, code=str(r["python_code"]),
                              index=int(i), tags=tags)


SOURCES = {
    "manimbench-train": lambda: manimbench("train"),
    "manimbench-test": lambda: manimbench("test"),
    "thanhkt": thanhkt,
    "generaleoley": generaleoley,
    "bespoke": bespoke,
}
