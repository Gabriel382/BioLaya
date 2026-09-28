from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class BiomedicalDocument:
    doc_id: str
    source: str
    text: str
    metadata: dict[str, Any]


def stable_partition(doc_id: str, validation_percent: int = 1) -> str:
    bucket = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def _is_allowed_pmc(row: dict[str, Any], allowed_licenses: set[str]) -> bool:
    if str(row.get("retracted", "no")).strip().lower() not in {"no", "false", "0", ""}:
        return False
    license_name = str(row.get("license", "")).strip().upper().replace("CC-BY", "CC BY")
    return any(license_name == allowed or license_name.startswith(allowed + " ") for allowed in allowed_licenses)


def _load_stream(repo: str, split: str):
    from datasets import load_dataset

    return load_dataset(repo, split=split, streaming=True)


def iter_pubmed(source_cfg: dict[str, Any], partition: str, validation_percent: int) -> Iterator[BiomedicalDocument]:
    ds = _load_stream(source_cfg["repo"], source_cfg.get("split", "train"))
    for row in ds:
        pmid = str(row.get("PMID", "")).strip()
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        text = "\n\n".join(x for x in (title, abstract) if x)
        if not text:
            continue
        doc_id = pmid or hashlib.sha1(text[:2048].encode("utf-8")).hexdigest()
        if stable_partition(doc_id, validation_percent) != partition:
            continue
        yield BiomedicalDocument(doc_id=doc_id, source="pubmed", text=text, metadata={"pmid": pmid})


def iter_pmc(source_cfg: dict[str, Any], partition: str, validation_percent: int) -> Iterator[BiomedicalDocument]:
    ds = _load_stream(source_cfg["repo"], source_cfg.get("split", "main"))
    allowed = {str(x).strip().upper() for x in source_cfg.get("allowed_licenses", ["CC0", "CC BY", "CC BY-SA"])}
    for row in ds:
        if source_cfg.get("filter_licenses", True) and not _is_allowed_pmc(row, allowed):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        pmid = str(row.get("pmid", "")).strip()
        accession = str(row.get("accession_id", "")).strip()
        doc_id = pmid if pmid and pmid != "0" else accession
        if not doc_id:
            doc_id = hashlib.sha1(text[:2048].encode("utf-8")).hexdigest()
        if stable_partition(doc_id, validation_percent) != partition:
            continue
        yield BiomedicalDocument(
            doc_id=doc_id,
            source="pmc",
            text=text,
            metadata={"pmid": pmid, "accession_id": accession, "license": row.get("license")},
        )


class BiomedicalMixture:
    """Deterministic weighted streaming mixture of PubMed and PMC."""

    def __init__(self, config: dict[str, Any], partition: str = "train", seed: int = 42):
        self.config = config
        self.partition = partition
        self.seed = seed

    def __iter__(self) -> Iterator[BiomedicalDocument]:
        corpus = self.config["corpus"]
        validation_percent = int(corpus.get("validation_percent", 1))
        sources = corpus["sources"]
        iterators: dict[str, Iterator[BiomedicalDocument]] = {}
        weights: dict[str, float] = {}
        for item in sources:
            name = str(item["name"])
            if name == "pubmed":
                iterator = iter_pubmed(item, self.partition, validation_percent)
            elif name == "pmc":
                iterator = iter_pmc(item, self.partition, validation_percent)
            else:
                raise KeyError(f"Unsupported corpus source: {name}")
            iterators[name] = iter(iterator)
            weights[name] = float(item.get("weight", 1.0))

        rng = random.Random(self.seed + (0 if self.partition == "train" else 100_000))
        active = dict(iterators)
        while active:
            names = list(active)
            probs = [weights[n] for n in names]
            name = rng.choices(names, weights=probs, k=1)[0]
            try:
                yield next(active[name])
            except StopIteration:
                del active[name]


def preflight_sources(config: dict[str, Any], partition: str = "train") -> list[dict[str, Any]]:
    """Read one usable document from every configured source before loading a large model."""
    corpus = config["corpus"]
    validation_percent = int(corpus.get("validation_percent", 1))
    rows = []
    for item in corpus["sources"]:
        name = item["name"]
        if name == "pubmed":
            iterator = iter_pubmed(item, partition, validation_percent)
        elif name == "pmc":
            iterator = iter_pmc(item, partition, validation_percent)
        else:
            raise KeyError(name)
        first = next(iter(iterator), None)
        if first is None:
            raise RuntimeError(f"Corpus preflight failed: source {name!r} yielded no usable {partition} document")
        rows.append({"source": name, "doc_id": first.doc_id, "chars": len(first.text)})
    return rows
