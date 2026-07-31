#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


def log_stage(message: str) -> None:
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[pnsearch] {message}", flush=True)


def resolve_deepspeed_config(
    path: str | Path, *, batch_size: int, gradient_accumulation: int, world_size: int
) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["train_micro_batch_size_per_gpu"] = batch_size
    config["gradient_accumulation_steps"] = gradient_accumulation
    config["train_batch_size"] = world_size * batch_size * gradient_accumulation
    return config


def model_load_deepspeed_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    # DeepSpeed communication is not initialized yet inside from_pretrained, so ZeRO Init sees
    # world_size=1. Trainer installs the true distributed batch configuration afterwards.
    result["train_batch_size"] = (
        int(result["train_micro_batch_size_per_gpu"])
        * int(result["gradient_accumulation_steps"])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for PN-Search reasoner or reranker")
    parser.add_argument("--task", choices=["reasoner", "reranker"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--validation", action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--deepspeed")
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    from pnsearch.training.formatting import reasoner_messages, reranker_messages

    files = {"train": args.train}
    if args.validation:
        files["validation"] = args.validation
    log_stage("loading training datasets")
    dataset = load_dataset("json", data_files=files)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatter = reranker_messages if args.task == "reranker" else reasoner_messages

    def encode(example):
        messages = formatter(example)
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        response_ids = full_ids[len(prompt_ids) :]
        if not response_ids:
            raise ValueError("chat template produced an empty supervised response")
        response_ids = response_ids[: args.max_length]
        prompt_budget = max(0, args.max_length - len(response_ids))
        if len(prompt_ids) > prompt_budget:
            # Preserve the system/query prefix and the most recent candidates/history suffix.
            prefix_size = min(prompt_budget // 3, len(prompt_ids))
            suffix_size = prompt_budget - prefix_size
            prompt_ids = prompt_ids[:prefix_size] + (
                prompt_ids[-suffix_size:] if suffix_size else []
            )
        input_ids = prompt_ids + response_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + response_ids,
        }

    tokenized = dataset.map(encode, remove_columns=dataset["train"].column_names)
    log_stage(
        f"encoded train={len(tokenized['train'])} validation="
        f"{len(tokenized.get('validation', []))} max_length={args.max_length}"
    )
    deepspeed_config = None
    resolved_deepspeed = args.deepspeed
    if args.deepspeed:
        from transformers.integrations import HfDeepSpeedConfig

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        resolved_deepspeed = resolve_deepspeed_config(
            args.deepspeed,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            world_size=world_size,
        )
        # Keep this object alive while loading so ZeRO-3 partitions the base model instead of
        # materializing one full 30B copy in every worker's CPU memory.
        deepspeed_config = HfDeepSpeedConfig(
            model_load_deepspeed_config(resolved_deepspeed)
        )
        log_stage(f"initialized ZeRO-3 for world_size={world_size}")
    log_stage(f"loading base model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    log_stage("base model loaded")
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        model.print_trainable_parameters()
    log_stage("LoRA adapters injected")
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps" if args.validation else "no",
        eval_steps=200,
        report_to="none",
        remove_unused_columns=False,
        deepspeed=resolved_deepspeed,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    log_stage(f"starting training max_steps={args.max_steps} epochs={args.epochs}")
    trainer.train()
    log_stage(f"saving adapter to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    log_stage("training and adapter save completed")
    _ = deepspeed_config


if __name__ == "__main__":
    main()
