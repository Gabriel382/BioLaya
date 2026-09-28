from __future__ import annotations

import inspect
import json
import math
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from biolaya.cloud.drive import atomic_copytree, restore_latest_checkpoint
from biolaya.corpus.biomedical_stream import preflight_sources
from biolaya.corpus.packing import PackedMLMDataset
from biolaya.utils.io import ensure_dir, write_json


@dataclass
class DAPTRuntime:
    device: str
    precision: str
    use_cpu: bool
    gradient_checkpointing: bool


def resolve_runtime(requested: str = "auto", mixed_precision: str = "auto", gradient_checkpointing: bool = True) -> DAPTRuntime:
    from biolaya.cloud.env import resolve_device
    import torch

    device = resolve_device(requested)
    if device == "cpu":
        return DAPTRuntime(device="cpu", precision="float32", use_cpu=True, gradient_checkpointing=gradient_checkpointing)
    precision = mixed_precision.lower()
    if precision == "auto":
        precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        precision = "float16"
    return DAPTRuntime(device="cuda", precision=precision, use_cpu=False, gradient_checkpointing=gradient_checkpointing)


def plan_dapt(config: dict[str, Any]) -> dict[str, Any]:
    train = config["training"]
    token_budget = int(config["corpus"]["token_budget"])
    seq = int(train["sequence_length"])
    batch = int(train.get("per_device_batch_size", 1))
    ga = int(train.get("gradient_accumulation_steps", 16))
    sequences = math.ceil(token_budget / seq)
    optimizer_steps = math.ceil(sequences / max(1, batch * ga))
    return {
        "run_name": config.get("run_name", "biolaya-dapt"),
        "base_model": config["model"]["base_model"],
        "token_budget": token_budget,
        "sequence_length": seq,
        "packed_sequences": sequences,
        "per_device_batch_size": batch,
        "gradient_accumulation_steps": ga,
        "effective_batch_sequences": batch * ga,
        "optimizer_steps": optimizer_steps,
        "sources": [{"name": s["name"], "weight": s.get("weight", 1.0), "repo": s["repo"]} for s in config["corpus"]["sources"]],
    }


def _compatible_training_arguments(desired: dict[str, Any], *, warmup_ratio: float) -> tuple[dict[str, Any], list[str]]:
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        out, skipped = dict(desired), []
    else:
        out = {k: v for k, v in desired.items() if k in params}
        skipped = sorted(set(desired) - set(out))
    if accepts_kwargs or "warmup_ratio" in params:
        out["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in params:
        # Transformers 5.x accepts a float fraction here as well.
        out["warmup_steps"] = warmup_ratio
    else:
        skipped.append("warmup_ratio/warmup_steps")
    return out, skipped


def _resolve_resume(resume: str | None, drive_run_dir: str | Path | None, local_output: Path) -> str | None:
    if not resume or resume.lower() in {"none", "false", "0"}:
        return None
    if resume.lower() != "auto":
        path = Path(resume)
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)
    if not drive_run_dir:
        candidates = sorted(local_output.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0]) if candidates else None
    restored = restore_latest_checkpoint(drive_run_dir, local_output)
    return str(restored) if restored else None


def train_dapt(
    config: dict[str, Any],
    *,
    device: str = "auto",
    resume: str | None = "auto",
    drive_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    train_cfg = config["training"]
    runtime = resolve_runtime(
        device,
        str(train_cfg.get("mixed_precision", "auto")),
        bool(train_cfg.get("gradient_checkpointing", True)),
    )
    plan = plan_dapt(config)
    output_dir = ensure_dir(config.get("output_dir", f"outputs/sprint2/{config.get('run_name', 'dapt')}"))

    print("BioLaya Sprint 2 runtime")
    print(f"  device:            {runtime.device}")
    if runtime.device == "cuda":
        print(f"  GPU:               {torch.cuda.get_device_name(0)}")
    print(f"  precision:         {runtime.precision}")
    print(f"  base model:        {plan['base_model']}")
    print(f"  token budget:      {plan['token_budget']:,}")
    print(f"  sequence length:   {plan['sequence_length']}")
    print(f"  optimizer steps:   {plan['optimizer_steps']:,}")

    if config.get("corpus", {}).get("preflight", True):
        print("BioLaya corpus preflight")
        for row in preflight_sources(config, "train"):
            print(f"  corpus OK: {row['source']:<7} chars={row['chars']:,} id={row['doc_id']}")

    tokenizer = AutoTokenizer.from_pretrained(plan["base_model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model = AutoModelForMaskedLM.from_pretrained(plan["base_model"])
    if runtime.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    train_ds = PackedMLMDataset(
        config,
        tokenizer,
        partition="train",
        token_budget=plan["token_budget"],
        sequence_length=plan["sequence_length"],
        seed=int(config.get("seed", 42)),
    )
    eval_budget = int(config["corpus"].get("validation_token_budget", min(1_000_000, max(plan["sequence_length"] * 64, plan["token_budget"] // 100))))
    eval_ds = PackedMLMDataset(
        config,
        tokenizer,
        partition="validation",
        token_budget=eval_budget,
        sequence_length=plan["sequence_length"],
        seed=int(config.get("seed", 42)) + 999,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=float(train_cfg.get("mlm_probability", 0.15)))

    desired = dict(
        output_dir=str(output_dir),
        max_steps=int(plan["optimizer_steps"]),
        per_device_train_batch_size=int(train_cfg.get("per_device_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=runtime.precision == "bfloat16",
        fp16=runtime.precision == "float16",
        gradient_checkpointing=runtime.gradient_checkpointing,
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        remove_unused_columns=False,
        use_cpu=runtime.use_cpu,
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        optim=train_cfg.get("optimizer", "adamw_torch"),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        run_name=config.get("run_name", "biolaya-dapt"),
    )
    compatible, skipped = _compatible_training_arguments(desired, warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)))
    if skipped:
        warnings.warn("Unsupported TrainingArguments skipped: " + ", ".join(skipped), RuntimeWarning, stacklevel=2)
    args = TrainingArguments(**compatible)

    callbacks = []
    if drive_run_dir:
        from biolaya.training.drive_callback import make_drive_backup_callback
        callbacks.append(make_drive_backup_callback(drive_run_dir, keep=int(train_cfg.get("drive_keep_checkpoints", 2))))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    resume_path = _resolve_resume(resume, drive_run_dir, output_dir)
    if resume_path:
        print(f"  resume checkpoint: {resume_path}")
    result = trainer.train(resume_from_checkpoint=resume_path)
    metrics = trainer.evaluate()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    manifest = {
        "format": "biolaya-sprint2-mlm-v1",
        "purpose": "Biomedical domain-adapted ModernBERT backbone for later BioLaya decision fine-tuning",
        "config": config,
        "plan": plan,
        "runtime": asdict(runtime),
        "train_metrics": result.metrics,
        "eval_metrics": metrics,
        "resume_from_checkpoint": resume_path,
    }
    write_json(final_dir / "training_manifest.json", manifest)
    write_json(output_dir / "training_manifest.json", manifest)
    if drive_run_dir:
        drive_run_dir = Path(drive_run_dir)
        atomic_copytree(final_dir, drive_run_dir / "final")
        write_json(drive_run_dir / "training_manifest.json", manifest)
        print(f"[Drive] final model saved: {drive_run_dir / 'final'}")
    print(f"BioLaya Sprint 2 complete: {final_dir}")
    return {"output_dir": str(output_dir), "final": str(final_dir), "metrics": metrics, "plan": plan}
