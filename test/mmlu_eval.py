#!/usr/bin/env python
"""
Lightweight MMLU evaluator for Qwen3 chat models.

Usage example:
  python mmlu_eval.py --model_name_or_path /path/to/ckpt --device cuda:0
"""

import argparse
import os
import random
import sys
import logging
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.modeling_qwen3_test_mmlu import Qwen3ForCausalLM

# Official 57 MMLU subjects.
MMLU_SUBJECTS: List[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

CHOICE_LETTERS = ["A", "B", "C", "D"]


def _get_first_available(example: Dict, keys: Sequence[str]):
    for k in keys:
        if k in example and example[k] is not None:
            return example[k]
    raise KeyError(f"Missing keys {keys} in example: {list(example.keys())}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on MMLU.")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="HF repo or local checkpoint to evaluate.",
    )
    parser.add_argument(
        "--dataset_name",
        default="cais/mmlu",
        help="Dataset repo/path. Expecting the MMLU (HendrycksTest) format.",
    )
    parser.add_argument(
        "--subjects",
        default="all",
        help="Comma-separated subject list or 'all' for the 57 default subjects.",
    )
    parser.add_argument("--few_shot", type=int, default=0, help="How many dev examples to prefix.")
    parser.add_argument(
        "--shot_split",
        default="dev",
        help="Split used for few-shot examples (typically 'dev' or 'validation').",
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        help="Split to evaluate on (typically 'test').",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--device", default=None, help="Device string, e.g., cuda:0 or cpu.")
    parser.add_argument("--bf16", action="store_true", help="Load model in bfloat16.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--mask_rate", type=float, default=0.0, help="Mask rate for massive value filtering (0.0 to 1.0)")
    parser.add_argument("--log_path", default=None, help="Optional log file path for summary.")
    return parser.parse_args()


def parse_answer(answer_field) -> int:
    """Map answer field to index 0-3."""
    if isinstance(answer_field, str):
        ans = answer_field.strip().upper()
        if ans in CHOICE_LETTERS:
            return CHOICE_LETTERS.index(ans)
        if ans.isdigit():
            return int(ans)
    return int(answer_field)


def format_example(example: Dict, include_answer: bool) -> str:
    question = _get_first_available(example, ["question", "input", "query", "prompt", "instruction"])
    choices = _get_first_available(example, ["choices", "options"])
    answer_raw = _get_first_available(example, ["answer", "answers", "target", "label"])

    answer_idx = parse_answer(answer_raw)
    lines = [f"Question: {question}"]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    if include_answer:
        lines.append(f"Answer: {CHOICE_LETTERS[answer_idx]}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_prompt(subject: str, shots: Sequence[Dict], example: Dict) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject}."
    blocks: List[str] = [header]
    if shots:
        blocks.append("\n\n".join(format_example(s, True) for s in shots))
    blocks.append(format_example(example, False))
    return "\n\n".join(blocks)


def score_choices(
    model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    choices: Sequence[str],
    device: torch.device,
    max_seq_length: int,
    mask_rate: float = 0.0,
) -> Tuple[List[float], List[int]]:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for multiple choice QA. "
            "Answer with the single letter of the correct option.",
        },
        {"role": "user", "content": prompt_text},
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    encoded_prompt = tokenizer(
        chat_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    prompt_len = encoded_prompt["input_ids"].size(1)

    scores: List[float] = []
    lengths: List[int] = []
    for choice in choices:
        candidate_text = chat_prompt + " " + choice
        encoded = tokenizer(
            candidate_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions = True, mask_rate=mask_rate)
            logits = output.logits

        # Log probabilities for tokens excluding the last position.
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]

        choice_len = input_ids.size(1) - prompt_len
        start = target_ids.size(1) - choice_len
        choice_log_probs = log_probs[:, start:start + choice_len, :]
        choice_targets = target_ids[:, start:start + choice_len]

        score = choice_log_probs.gather(2, choice_targets.unsqueeze(-1)).squeeze(-1).sum()
        scores.append(score.item())
        lengths.append(choice_len)

    return scores, lengths


def load_split(dataset_name: str, subject: str, split: str):
    candidates = [split]
    if split == "dev":
        candidates += ["validation", "val"]
    elif split in {"validation", "val"}:
        candidates += ["dev"]

    last_err = None
    for cand in candidates:
        try:
            return load_dataset(dataset_name, subject, split=cand)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"Could not load split {split} for {dataset_name}/{subject}")


def evaluate_subject(
    model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    subject: str,
    few_shot: int,
    shot_split: str,
    eval_split: str,
    device: torch.device,
    max_seq_length: int,
    max_samples: int,
    rng: random.Random,
    mask_rate: float = 0.0,
) -> Tuple[float, int, int, float, int]:
    dev_set = load_split(dataset_name, subject, shot_split) if few_shot > 0 else []
    eval_set = load_split(dataset_name, subject, eval_split)

    if max_samples:
        eval_set = eval_set.select(range(min(max_samples, len(eval_set))))

    shots: List[Dict] = []
    if few_shot > 0 and len(dev_set) > 0:
        indices = list(range(len(dev_set)))
        rng.shuffle(indices)
        shots = [dev_set[i] for i in indices[:few_shot]]

    correct = 0
    total = 0
    nll_sum = 0.0
    tok_count = 0
    for example in tqdm(eval_set, desc=subject):
        prompt = build_prompt(subject, shots, example)
        scores, lengths = score_choices(model, tokenizer, prompt, example["choices"], device, max_seq_length, mask_rate)
        pred = int(torch.tensor(scores).argmax().item())
        gold = parse_answer(example["answer"])
        correct += int(pred == gold)
        total += 1
        # Accumulate NLL for perplexity (use gold choice log prob).
        gold_len = lengths[gold]
        if gold_len > 0:
            nll_sum += -scores[gold]
            tok_count += gold_len

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total, nll_sum, tok_count


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(log_path: str | None) -> logging.Logger:
    logger = logging.getLogger("mmlu")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def main() -> None:
    args = parse_args()

    logger = setup_logger(args.log_path)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map=None,
    ).to(device)


    # model = LlamaForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     trust_remote_code=False,
    #     torch_dtype=torch.bfloat16 if args.bf16 else None,
    #     device_map=None,
    # ).to(device)

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16 if args.bf16 else None,
    #     device_map=None,
    # ).to(device)



    # layers = model.model.layers
    # layer1 = model_.model.layers
    # layers[6].post_attention_layernorm = layer1[6].post_attention_layernorm
    # layers[6].mlp = layer1[6].mlp
    model.eval()

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects != "all" else MMLU_SUBJECTS

    total_correct = 0
    total_seen = 0

    total_nll = 0.0
    total_tokens = 0

    for subject in subjects:
        acc, correct, count, nll_sum, tok_count = evaluate_subject(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            subject=subject,
            few_shot=args.few_shot,
            shot_split=args.shot_split,
            eval_split=args.eval_split,
            device=device,
            max_seq_length=args.max_seq_length,
            max_samples=args.max_samples,
            rng=rng,
            mask_rate=args.mask_rate,
        )
        total_correct += correct
        total_seen += count
        total_nll += nll_sum
        total_tokens += tok_count
        subj_ppl = float("inf") if tok_count == 0 else torch.exp(torch.tensor(nll_sum / tok_count)).item()
        print(f"{subject}: {acc * 100:.2f}% ({count} samples), ppl={subj_ppl:.2f}")

    overall = total_correct / total_seen if total_seen > 0 else 0.0
    overall_ppl = float("inf") if total_tokens == 0 else torch.exp(torch.tensor(total_nll / total_tokens)).item()
    summary = f"ft01 Overall: {overall * 100:.2f}% across {total_seen} samples, ppl={overall_ppl:.2f} 0.75 our"
    print(summary)
    logger.info(summary)


if __name__ == "__main__":

    # path = "/common/home/zs618/hidden_sink/checkpoints/qwen3-sft/qwen3-4b-6layer-hid-100k/global_step_580/extra_state_world_size_2_rank_1.pt"

    # obj = torch.load(path, map_location="cpu", weights_only=False)

    # def scan(obj, prefix=""):
    #     if isinstance(obj, dict):
    #         for k, v in obj.items():
    #             scan(v, prefix + k + ".")
    #     elif isinstance(obj, list):
    #         for idx, v in enumerate(obj):
    #             scan(v, prefix + f"[{idx}].")
    #     elif torch.is_tensor(obj):
    #         # 只打印 gate_proj 部分（你关心的）
    #         if "gate_proj" in prefix:
    #             print(prefix, obj.shape)

    # scan(obj)
    # print(ssd)
    main()
