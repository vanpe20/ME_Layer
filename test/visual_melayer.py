import argparse
import builtins
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_SAVE_PATH = REPO_ROOT / "visual_res" / "me_layer_l2_by_dim.png"
DEFAULT_PROMPT = "My favorite color is Dark Blue."

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modeling_llama import LlamaForCausalLM
from modeling_mistral import MistralForCausalLM
from modeling_phi3 import Phi3ForCausalLM
from modeling_qwen2 import Qwen2ForCausalLM
from modeling_qwen3 import Qwen3ForCausalLM


MODEL_CLASSES = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "phi3": Phi3ForCausalLM,
    "mistral": MistralForCausalLM,
    "llama": LlamaForCausalLM,
}


def infer_model_class(model_name_or_path: str):
    name = model_name_or_path.lower()
    if "qwen3" in name:
        return "qwen3", Qwen3ForCausalLM
    if "qwen2" in name or "qwen2.5" in name:
        return "qwen2", Qwen2ForCausalLM
    if "phi-3" in name or "phi3" in name:
        return "phi3", Phi3ForCausalLM
    if "mistral" in name:
        return "mistral", MistralForCausalLM
    if "llama" in name:
        return "llama", LlamaForCausalLM
    raise ValueError(
        "Cannot infer model class from --model. Model name/path should contain one of: "
        + ", ".join(MODEL_CLASSES)
    )


def load_model(model_name_or_path: str, device: str):
    model_family, model_cls = infer_model_class(model_name_or_path)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model_cls.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model, tokenizer, model_family


@contextmanager
def suppress_debug_plotting():
    original_breakpoint = builtins.breakpoint
    original_pybreakpoint = os.environ.get("PYTHONBREAKPOINT")
    patched = {}

    def noop(*args, **kwargs):
        return None

    for name in (
        "figure",
        "plot",
        "title",
        "xlabel",
        "ylabel",
        "grid",
        "savefig",
        "close",
        "tight_layout",
        "legend",
        "fill_between",
    ):
        patched[name] = getattr(plt, name)
        setattr(plt, name, noop)

    builtins.breakpoint = noop
    os.environ["PYTHONBREAKPOINT"] = "0"
    try:
        yield
    finally:
        builtins.breakpoint = original_breakpoint
        if original_pybreakpoint is None:
            os.environ.pop("PYTHONBREAKPOINT", None)
        else:
            os.environ["PYTHONBREAKPOINT"] = original_pybreakpoint
        for name, fn in patched.items():
            setattr(plt, name, fn)


def collect_token0_dim_l2_matrix(model, tokenizer, device: str) -> torch.Tensor:
    layer_outputs = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            layer_outputs.append((layer_idx, hidden.detach().float().cpu()))

        return hook

    handles = [
        layer.register_forward_hook(make_hook(layer_idx))
        for layer_idx, layer in enumerate(model.model.layers)
    ]

    inputs = tokenizer(DEFAULT_PROMPT, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        with torch.no_grad(), suppress_debug_plotting():
            model.model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if not layer_outputs:
        raise RuntimeError("No decoder layer outputs were collected.")

    rows = []
    for _layer_idx, hidden in sorted(layer_outputs, key=lambda x: x[0]):
        token0 = hidden[:, 0, :]  # [B, D]
        dim_l2 = torch.linalg.vector_norm(token0, dim=0)  # [D]
        rows.append(dim_l2)

    return torch.stack(rows, dim=0)  # [L, D]


def plot_layer_dim_l2_3d(matrix: torch.Tensor, save_path: Path):
    z = matrix.float().cpu().numpy()
    num_layers, hidden_dim = z.shape
    hidden_dims = np.arange(hidden_dim)
    layer_ids = np.arange(num_layers)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.4, 5.0))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for layer_idx in layer_ids:
        ax.plot(
            hidden_dims,
            np.full(hidden_dim, layer_idx),
            z[layer_idx],
            color="#6fa8dc",
            linewidth=0.75,
            alpha=0.78,
        )

    ax.set_title("ME Layer", fontsize=15, fontweight="bold", fontstyle="italic", pad=8)
    ax.text2D(0.03, 0.95, "Output of Decoder Layer", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Hidden Dim", labelpad=7, fontsize=11, fontweight="bold")
    ax.set_ylabel("Layer Idx", labelpad=7, fontsize=11, fontweight="bold")
    ax.set_zlabel("L2 Norm", labelpad=7, fontsize=11, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, linewidth=0.45, alpha=0.4)
    ax.view_init(elev=22, azim=48)
    ax.set_box_aspect((2.1, 1.0, 0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot token0 hidden-dimension L2 norms for every decoder layer."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HF model id or local checkpoint path.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer, model_family = load_model(args.model, args.device)
    matrix = collect_token0_dim_l2_matrix(model, tokenizer, args.device)
    plot_layer_dim_l2_3d(matrix, args.save_path)

    print(f"model_family={model_family}")
    print(f"matrix_shape=[layers={matrix.shape[0]}, hidden_dim={matrix.shape[1]}]")
    print(f"saved={args.save_path}")


if __name__ == "__main__":
    main()
