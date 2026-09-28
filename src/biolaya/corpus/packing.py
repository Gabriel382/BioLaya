from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from biolaya.corpus.biomedical_stream import BiomedicalMixture


class PackedMLMDataset(IterableDataset):
    """Stream biomedical documents into fixed-length token blocks for MLM."""

    def __init__(
        self,
        config: dict,
        tokenizer,
        *,
        partition: str,
        token_budget: int,
        sequence_length: int,
        seed: int,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.partition = partition
        self.token_budget = int(token_budget)
        self.sequence_length = int(sequence_length)
        self.seed = int(seed)

    @property
    def num_sequences(self) -> int:
        return math.ceil(self.token_budget / self.sequence_length)

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        separator = self.tokenizer.sep_token_id
        if separator is None:
            separator = self.tokenizer.eos_token_id
        if separator is None:
            separator = self.tokenizer.pad_token_id
        if separator is None:
            raise RuntimeError("Tokenizer needs sep/eos/pad token for packed biomedical DAPT")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = separator

        buffer: list[int] = []
        used_tokens = 0
        stream = BiomedicalMixture(self.config, partition=self.partition, seed=self.seed)
        for doc in stream:
            ids = self.tokenizer(doc.text, add_special_tokens=False, truncation=False)["input_ids"]
            if not ids:
                continue
            buffer.extend(ids)
            buffer.append(int(separator))
            while used_tokens < self.token_budget:
                remaining = self.token_budget - used_tokens
                real_len = min(self.sequence_length, remaining)
                if len(buffer) < real_len:
                    break
                block = buffer[:real_len]
                del buffer[:real_len]
                attention = [1] * real_len
                used_tokens += real_len
                if real_len < self.sequence_length:
                    missing = self.sequence_length - real_len
                    block += [int(pad_id)] * missing
                    attention += [0] * missing
                yield {"input_ids": block, "attention_mask": attention}
                if used_tokens >= self.token_budget:
                    return

        if used_tokens < self.token_budget and buffer:
            remaining = self.token_budget - used_tokens
            real_len = min(len(buffer), self.sequence_length, remaining)
            block = buffer[:real_len]
            attention = [1] * real_len
            missing = self.sequence_length - real_len
            block += [int(pad_id)] * missing
            attention += [0] * missing
            yield {"input_ids": block, "attention_mask": attention}
