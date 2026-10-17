"""Explicit single-GPU QLoRA job; review gate runs before optional GPU imports."""

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.training.artifacts import code_provenance, file_hash, write_json
from ai_service.training.dataset import DatasetSplit, load_dataset
from ai_service.training.export import build_example
from ai_service.training.qwen_config import QwenConfig
from ai_service.training.qwen_data import IGNORE_LABEL, tokenize_example


def train(directory: Path, output: Path, config: QwenConfig) -> dict[str, Any]:
    loaded = load_dataset(directory, require_approved=True)
    examples = {
        split: [
            build_example(sample, loaded.snapshots[sample.snapshot_id])
            for sample in loaded.samples
            if sample.split == split
        ]
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION)
    }
    if not all(examples.values()):
        raise ValueError("TRAIN_AND_VALIDATION_REQUIRED")
    if output.exists():
        raise FileExistsError(output)
    # Keep these packages out of server import/lifecycle and normal CI installs.
    torch = import_module("torch")
    transformers = import_module("transformers")
    datasets = import_module("datasets")
    peft = import_module("peft")
    trl = import_module("trl")
    if not torch.cuda.is_available():
        raise ValueError("QLORA_REQUIRES_CUDA")
    transformers.set_seed(config.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id, revision=config.model_revision
    )
    tokenized = {
        split: [
            tokenize_example(example, tokenizer, max_length=config.max_length)
            for example in rows
        ]
        for split, rows in examples.items()
    }
    output.mkdir(parents=True, exist_ok=False)
    provenance = code_provenance()
    run = {
        "schema_version": 1,
        "config": config.model_dump(mode="json"),
        "dataset": loaded.manifest.model_dump(mode="json"),
        "code": provenance,
        "request_hashes": {
            example["sample_id"]: example["request_hash"]
            for rows in examples.values()
            for example in rows
        },
        "chat_template_hash": canonical_hash(tokenizer.chat_template),
        "compute_dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(output / "run.json", run)
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=dtype,
    )
    model.config.use_cache = False
    model = peft.prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    arguments = trl.SFTConfig(
        output_dir=str(output / "checkpoints"),
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=1,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        push_to_hub=False,
    )
    trainer = trl.SFTTrainer(
        model=model,
        args=arguments,
        processing_class=tokenizer,
        train_dataset=datasets.Dataset.from_list(tokenized[DatasetSplit.TRAIN]),
        eval_dataset=datasets.Dataset.from_list(tokenized[DatasetSplit.VALIDATION]),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, label_pad_token_id=IGNORE_LABEL
        ),
    )
    result = trainer.train()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    adapter_files = {
        path.name: file_hash(path)
        for path in (output / "adapter").iterdir()
        if path.is_file()
    }
    manifest = {
        **run,
        "adapter_files": adapter_files,
        "metrics": result.metrics,
        "selection": "minimum validation eval_loss; not a semantic promotion gate",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        train(
            args.directory,
            args.output,
            QwenConfig.model_validate_json(args.config.read_bytes()),
        )
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"TRAINING_FAILED: {error}\n")
    print(
        f"Adapter saved to {args.output / 'adapter'}; no runtime promotion performed."
    )


if __name__ == "__main__":
    main()
