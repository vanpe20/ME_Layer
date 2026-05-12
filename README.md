<div align="center">
  <h1 style="font-size: 32px; font-weight: bold;"> A Single Layer to Explain Them All: Understanding Massive Values in Large Language Models </h1>

  <br>
  <a href="https://arxiv.org/pdf/2605.08504">
    <img src="https://img.shields.io/badge/ArXiv-WeMask-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://huggingface.co/DarkBluee/WeMask">
    <img src="https://img.shields.io/badge/🤗 huggingface-Model-purple" alt="checkpoint">
  </a>
  <a href="https://vanpe20.github.io/ME-Layer.github.io/">
    <img src="https://img.shields.io/badge/-HomePage-black?logo=github" alt="checkpoint">
  </a>
</div>




## Quick Start

### Environment Setup

```bash
conda create -n me-layer python=3.10 -y
conda activate me-layer
cd /common/home/zs618/hidden_sink/ME_Layer
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


### Data

The training data used in this project includes:

- **FLAN**: [https://huggingface.co/datasets/Open-Orca/FLAN]
- **GSM8K-CoT**: [https://huggingface.co/datasets/ankner/gsm8k-CoT]
- **GSM8K-CoT-verl**: You can find it at [ME_Layer/data](./data)

There are also several test datasets, including AIME24, Math500, MMLU, and XSTest. We provide some of them in [ME_Layer/data](./data)

### Visualize MELayer

Use the visualization script to plot the L2 norm of token-0 hidden dimensions across decoder layers. For Qwen3-4B:

```bash

MODEL=Qwen/Qwen3-4B \
DEVICE=cuda:0 \
SAVE_PATH=./visual_res/qwen3_4b_melayer_l2.png \
bash scripts/run_visual_melayer.sh
```

### Training

To start training, you should download the training data first.

Start SFT training:

```bash
bash scripts/run_verl_sft_qwen3_math.sh
```

Start GRPO training on GSM8K-CoT:

```bash
bash scripts/run_verl_grpo_qwen3_4b_gsm8k.sh
```

### Testing

You can change the `MASK_RATE` in the bash to test the impact of mask rate on performance.

Run Math-500 evaluation:

```bash
bash scripts/test_math500.sh
```

Run GSM8K evaluation:

```bash
bash scripts/test_gsm8k.sh
```

## Citation

If you think our research is helpful, please cite with

```bibtex
@article{me_layer_2026,
  title={A Single Layer to Explain Them All: Understanding Massive Values in Large Language Models},
  author={Your Name and Co-authors},
  journal={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```
