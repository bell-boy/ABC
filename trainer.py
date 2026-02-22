import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ImageFile import ImageFile
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
MAX_SEQUENCE_LENGTH = 2048
PROMPT_TEXT = "Convert the following sheet music into ABC notation:"
EVAL_SPLIT_RATIO = 0.1


def format_train_sample(images: List[ImageFile], abc_notation: str) -> Dict[str, Any]:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_TEXT},
                *[{"type": "image", "data": img} for img in images],
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": abc_notation}],
        },
    ]
    return {"messages": conversation}


def format_eval_sample(images: List[ImageFile]) -> Dict[str, Any]:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_TEXT},
                *[{"type": "image", "data": img} for img in images],
            ],
        }
    ]
    return {"messages": conversation}


def _load_metadata_records(path: Path) -> List[Dict[str, Any]]:
    with (path / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if isinstance(metadata, dict):
        records = metadata.get("scores", [])
    else:
        records = metadata

    if not isinstance(records, list):
        raise ValueError(
            "metadata.json must contain a list of records or {'scores': [...]}"
        )
    return records


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    records = _load_metadata_records(path)

    dataset: List[Dict[str, Any]] = []
    for item in records:
        score_id = str(item["score_id"])
        saves = int(item.get("saves", 0))
        pages = int(item["pages"])
        example_path = path / score_id
        images: List[ImageFile] = []
        for i in range(1, pages + 1):
            image_path = example_path / f"page_{i}.png"
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        abc_notation = (example_path / f"{score_id}.abc").read_text(encoding="utf-8")
        dataset.append(
            {
                "score_id": score_id,
                "saves": saves,
                "images": images,
                "abc_notation": abc_notation,
            }
        )
    return dataset


def split_dataset(
    examples: List[Dict[str, Any]],
    eval_split_ratio: float = EVAL_SPLIT_RATIO,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0 <= eval_split_ratio < 1:
        raise ValueError("eval_split_ratio must be in the range [0, 1).")
    if not examples:
        return [], []

    ranked_examples = sorted(
        examples,
        key=lambda example: (-int(example.get("saves", 0)), str(example["score_id"])),
    )

    eval_size = int(len(ranked_examples) * eval_split_ratio)
    if len(ranked_examples) >= 2:
        eval_size = max(1, eval_size)
        eval_size = min(eval_size, len(ranked_examples) - 1)

    eval_examples = ranked_examples[:eval_size]
    train_examples = ranked_examples[eval_size:]
    return train_examples, eval_examples


def build_train_dataset(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        format_train_sample(example["images"], example["abc_notation"])
        for example in examples
    ]


def levenshtein_distance(source: str, target: str) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current_row = [i]
        for j, target_char in enumerate(target, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (source_char != target_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _get_text_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def _decode_tokens(tokenizer: Any, token_ids: List[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    text_tokenizer = _get_text_tokenizer(tokenizer)
    return text_tokenizer.decode(token_ids, skip_special_tokens=True)


def _find_last_subsequence(
    sequence: List[int], subsequence: List[int]
) -> Optional[Tuple[int, int]]:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    return None


def eval(
    model: Any,
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    max_new_tokens: int = 512,
) -> Dict[str, float]:
    FastVisionModel.for_inference(model)
    model.eval()

    collator = UnslothVisionDataCollator(model, tokenizer)
    device = next(model.parameters()).device
    text_tokenizer = _get_text_tokenizer(tokenizer)

    distances: List[int] = []
    nll_sum = 0.0
    nll_token_count = 0

    for example in examples:
        reference_text = example["abc_notation"]

        prompt_sample = format_eval_sample(example["images"])
        prompt_batch = _to_device(collator([prompt_sample]), device)
        prompt_inputs = {k: v for k, v in prompt_batch.items() if k != "labels"}
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = model.generate(
                **prompt_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_suffix = generated_ids[0, prompt_len:].detach().cpu().tolist()
        prediction_text = _decode_tokens(tokenizer, generated_suffix).strip()
        distances.append(levenshtein_distance(prediction_text, reference_text))

        train_sample = format_train_sample(example["images"], reference_text)
        train_batch = _to_device(collator([train_sample]), device)
        input_ids = train_batch["input_ids"]
        full_token_ids = input_ids[0].detach().cpu().tolist()
        text_token_ids = text_tokenizer(reference_text, add_special_tokens=False)[
            "input_ids"
        ]
        text_span = _find_last_subsequence(full_token_ids, text_token_ids)

        if text_span is None:
            continue
        text_start, text_end = text_span

        with torch.inference_mode():
            outputs = model(**{k: v for k, v in train_batch.items() if k != "labels"})

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)

        positions = torch.arange(1, input_ids.shape[1], device=device).unsqueeze(0)
        text_mask = (positions >= text_start) & (positions < text_end)
        nll_sum += token_losses[text_mask].sum().item()
        nll_token_count += int(text_mask.sum().item())

    mean_distance = (
        float(sum(distances) / len(distances)) if distances else float("nan")
    )
    mean_nll = (nll_sum / nll_token_count) if nll_token_count else float("nan")

    return {
        "num_examples": float(len(examples)),
        "mean_levenshtein_distance": mean_distance,
        "mean_text_nll": mean_nll,
        "num_text_tokens": float(nll_token_count),
    }


def get_model_and_tokenizer():
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        load_in_4bit=True,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def main():
    model, tokenizer = get_model_and_tokenizer()
    examples = load_dataset(Path("data"))
    train_examples, eval_examples = split_dataset(examples)

    if not train_examples:
        raise ValueError("No training examples were found after splitting.")

    print(
        f"Loaded {len(train_examples)} train examples and "
        f"{len(eval_examples)} eval examples (top {int(EVAL_SPLIT_RATIO * 100)}% by saves)."
    )
    train_dataset = build_train_dataset(train_examples)

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_SEQUENCE_LENGTH,
        ),
    )

    trainer.train()

    if not eval_examples:
        print("No evaluation examples available after split; skipping evaluation.")
        return

    metrics = eval(model, tokenizer, eval_examples)
    print(
        "Eval metrics | "
        f"mean_levenshtein_distance={metrics['mean_levenshtein_distance']:.4f}, "
        f"mean_text_nll={metrics['mean_text_nll']:.6f}, "
        f"num_examples={int(metrics['num_examples'])}, "
        f"num_text_tokens={int(metrics['num_text_tokens'])}"
    )


if __name__ == "__main__":
    main()
