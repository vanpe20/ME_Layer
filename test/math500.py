import argparse
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from math_equivalence import is_equiv
from src.modeling_qwen3_test_math import Qwen3ForCausalLM


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_PATH = os.path.join(REPO_ROOT, "res_log_math500", "math500_metrics.log")
PROMPT_PREFIX = "Please reason step by step and put your final answer within \\boxed{{}}.\nQuestion: "


def default_data_path() -> str:
    candidates = [
        os.path.join(REPO_ROOT, "dataset", "Math-500", "test.jsonl"),
        os.path.join(REPO_ROOT, "data", "Math-500", "test.jsonl"),
        os.path.join(os.path.dirname(REPO_ROOT), "dataset", "Math-500", "test.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


DEFAULT_DATA_PATH = default_data_path()


def load_records(data_path: str):
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
        return df.to_dict(orient="records")

    with open(data_path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def get_question(item: dict) -> str:
    question = item.get("problem") or item.get("question")
    if not question:
        raise KeyError(f"Missing question in item keys: {list(item.keys())}")
    return question


def get_answer(item: dict) -> str:
    answer = item.get("answer")
    if answer is None:
        raise KeyError(f"Missing answer in item keys: {list(item.keys())}")
    return str(answer)


def setup_logger(log_path: str | None) -> logging.Logger:
    logger = logging.getLogger("math500")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def math_generate(model, tokenizer, device, data_path, max_new_tokens, batch_size, mask_rate, logger):
    dataset = load_records(data_path)
    total_lines = len(dataset)
    ans_list = []

    for i in tqdm(range(0, total_lines, batch_size), desc="Batch inference"):
        batch = dataset[i:i + batch_size]

        prompts = [PROMPT_PREFIX + get_question(item) for item in batch]
        answers = [get_answer(item) for item in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = inputs["input_ids"]
        attn_masks = inputs["attention_mask"]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attn_masks,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                mask_rate=mask_rate,
            )

        output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for out_text, gold in zip(output_texts, answers):
            ans_list.append(is_equiv(out_text, gold))

    total = len(ans_list)
    correct = int(sum(ans_list))
    true_ratio = correct / total if total else 0.0
    logger.info("Math-500 accuracy: %.4f (%d/%d) with mask_rate=%.3f", true_ratio, correct, total, mask_rate)
    return true_ratio


def main():
    parser = argparse.ArgumentParser(description="Math-500 evaluation")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-14B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--log_path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mask_rate", type=float, default=0.0, help="Mask rate for massive value filtering (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", help="Load model in bfloat16.")
    args = parser.parse_args()

    logger = setup_logger(args.log_path)
    if args.seed is not None:
        set_seed(args.seed)
        logger.info("seed=%d", args.seed)

    logger.info(
        "model_name_or_path=%s device=%s data_path=%s batch_size=%d max_new_tokens=%d mask_rate=%.3f",
        args.model_name_or_path,
        args.device,
        args.data_path,
        args.batch_size,
        args.max_new_tokens,
        args.mask_rate,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        output_attentions=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    ).to(args.device)

    math_generate(model, tokenizer, args.device, args.data_path, args.max_new_tokens, args.batch_size, args.mask_rate, logger)


if __name__ == "__main__":
    main()
