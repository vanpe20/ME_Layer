import numpy as np
import torch
import torch.nn as nn
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.modeling_qwen3_test_math import Qwen3ForCausalLM
# from modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_equivalence import is_equiv
import logging
import os
import time
import copy
import argparse
import random


try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


DEFAULT_DATA_PATH = "/common/home/zs618/hidden_sink/dataset/AIME24/train-00000-of-00001.parquet"
DEFAULT_LOG_PATH = "/common/home/zs618/hidden_sink/res_log_aime24/aime2024_metrics.log"
PROMPT_PREFIX = "Please reason step by step and put your final answer within \\boxed{{}}."


def visualize_attention_sink(attentions, save_dir="attention_sink", prefix="sample_0"):
    if not attentions:
        return

    os.makedirs(save_dir, exist_ok=True)
    last_step_attn = attentions[-1]
    if isinstance(last_step_attn, (list, tuple)):
        layer_attentions = last_step_attn
    else:
        layer_attentions = (last_step_attn,)

    for layer_idx, layer_attn in enumerate(layer_attentions):
        if isinstance(layer_attn, torch.Tensor):
            attn_tensor = layer_attn.detach()
        else:
            attn_tensor = torch.tensor(layer_attn)

        if attn_tensor.ndim == 4:
            attn_tensor = attn_tensor.mean(dim=1)
        if attn_tensor.ndim == 3:
            attn_tensor = attn_tensor[0]

        if attn_tensor.ndim != 2 or attn_tensor.size(0) == 0:
            continue

        sink_profile = attn_tensor[-1].cpu().numpy()
        first_token_weight = float(sink_profile[0]) if sink_profile.size > 0 else float("nan")
        print(f"[Attention Sink] {prefix} layer {layer_idx}: attention to token 0 = {first_token_weight:.4f}")

        plt.figure()
        plt.plot(range(len(sink_profile)), sink_profile)
        plt.xlabel("Key Token Index")
        plt.ylabel("Attention Weight")
        plt.title(f"{prefix} Layer {layer_idx} attention sink profile")
        plt.tight_layout()
        output_path = os.path.join(save_dir, "attention_sink.png")
        plt.savefig(output_path)
        breakpoint()
        plt.close()


def setup_logger(log_path):
    logger = logging.getLogger("aime2024")
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


def _read_parquet_rows(data_path):
    if pd is not None:
        df = pd.read_parquet(data_path)
        return df.to_dict(orient="records")

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing parquet reader. Install pandas or pyarrow to load AIME24 data."
        ) from exc

    table = pq.read_table(data_path)
    return table.to_pylist()


def _pick_first(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _extract_problem_answer(row):
    problem = _pick_first(row, ["problem", "question", "prompt", "input", "text"])
    answer = _pick_first(
        row,
        [
            "answer",
            "final_answer",
            "short_answer",
            "solution",
            "target",
            "response",
            "output",
            "label",
        ],
    )
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else None
    if isinstance(answer, dict):
        answer = _pick_first(answer, ["answer", "final", "value"])
    if problem is None or answer is None:
        return None, None
    return str(problem), str(answer)


def _load_aime_dataset(data_path, logger):
    rows = _read_parquet_rows(data_path)
    pairs = []
    skipped = 0
    for row in rows:
        problem, answer = _extract_problem_answer(row)
        if problem is None or answer is None:
            skipped += 1
            continue
        pairs.append((problem, answer))

    if skipped:
        logger.info("Skipped %d rows with missing problem/answer fields.", skipped)
    return pairs


def math_generate(model, tokenizer, device, data_path, max_new_tokens, batch_size, mask_rate, logger):
    dataset = _load_aime_dataset(data_path, logger)
    total_lines = len(dataset)
    if total_lines == 0:
        logger.warning("No valid samples found in %s", data_path)
        return 0.0

    ans_list = []
    for i in tqdm(range(0, total_lines, batch_size), desc="Batch inference"):
        batch = dataset[i:i + batch_size]

        prompts = [PROMPT_PREFIX + problem for problem, _ in batch]
        answers = [gold for _, gold in batch]

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
            isright = is_equiv(out_text, gold)
            ans_list.append(isright)

    total = len(ans_list)
    correct = int(sum(ans_list))
    true_ratio = correct / total if total else 0.0
    logger.info("AIME2024 accuracy: %.4f (%d/%d) with mask_rate=%.3f", true_ratio, correct, total, mask_rate)
    return true_ratio


def main():
    parser = argparse.ArgumentParser(description="AIME2024 evaluation")
    parser.add_argument("--model_name_or_path", default="/common/users/zs618/sink_model/qwen3-4b-rl_math_60")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--log_path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=512)
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
