<h1 align="center"> Flow-GRPO:<br>Training Flow Matching Models via Online RL </h1>
<div align="center">
  <a href='https://arxiv.org/abs/2505.05470'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://gongyeliu.github.io/Flow-GRPO/'><img src='https://img.shields.io/badge/Visualization-green?logo=github'></a> &nbsp;
  <a href="https://github.com/yifan123/flow_grpo"><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp; 
  <a href='https://huggingface.co/collections/jieliu/sd35m-flowgrpo-68298ec27a27af64b0654120'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <a href='https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO'><img src='https://img.shields.io/badge/Demo-blue?logo=huggingface'></a> &nbsp;
</div>

## 📝 Updates
- __[2025.05.15]__: 🔥We showcase image examples from three tasks and their training evolution at https://gongyeliu.github.io/Flow-GRPO. Check them out!
- __[2025.05.13]__: 🔥We now provide an online demo for all three tasks at https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO. You're welcome to try it out!

## 🤗 Model
| Task    | Model |
| -------- | -------- |
| GenEval     | [🤗GenEval](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-GenEval) |
| Text Rendering     | [🤗Text](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text) |
| Human Preference Alignment     | [🤗PickScore](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-PickScore) |

## 🚀 Quick Started
### 1. Environment Set Up
Clone this repository and install packages.
```bash
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo
conda create -n flow_grpo python=3.10.16
pip install -e .
```
### 2. Reward Preparation
The steps above only install the current repository. Since each reward model may rely on different versions, combining them in one Conda environment can cause version conflicts. To avoid this, we adopt a remote server setup inspired by ddpo-pytorch. You only need to install the specific reward model you plan to use.

#### GenEval
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server).

#### OCR
Please install paddle-ocr:
```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```
Then, pre-download the model using the Python command line:
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
```

#### Pickscore
PickScore requires no additional installation.

#### DeQA
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server).

#### UnifiedReward
We use sglang to deploy the reward service, and we also recommend using sglang or vllm for deploying VLM-based reward models.
After installing sglang, please run the following command to launch UnifiedReward:

```bash
python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85
```

### 3. Start Training
#### GRPO
Single-node training:
```bash
bash scripts/single_node/grpo.sh
```
Multi-node training:
```bash
# Master node
bash scripts/multi_node/main.sh
# Other nodes
bash scripts/multi_node/main1.sh
bash scripts/multi_node/main2.sh
bash scripts/multi_node/main3.sh
```
#### DPO / OnlineDPO / SFT / OnlineSFT
 Single-node training:
```bash
bash scripts/single_node/dpo.sh
bash scripts/single_node/sft.sh
```
Multi-node training:

Please update the entry Python script and config file names in the `scripts/multi_node` bash file.


## 🏁 Multi Reward Training
For multi-reward settings, you can pass in a dictionary where each key is a reward name and the corresponding value is its weight.
For example:

```python
{
    "pickscore": 0.5,
    "ocr": 0.2,
    "aesthetic": 0.3
}
```

This means the final reward is a weighted sum of the individual rewards.

The following reward models are currently supported:
* **Geneval** evaluates T2I models on complex compositional prompts.
* **OCR** provides an OCR-based reward.
* **PickScore** is a general-purpose T2I reward model trained on human preferences.
* **[DeQA](https://github.com/zhiyuanyou/DeQA-Score)** is a multimodal LLM-based image quality assessment model that measures the impact of distortions and texture damage on perceived quality.
* **ImageReward** is a general-purpose T2I reward model capturing text-image alignment, visual fidelity, and safety.
* **QwenVL** is an experimental reward model using prompt engineering.
* **Aesthetic** is a CLIP-based linear regressor predicting image aesthetic scores.
* **JPEG\_Compressibility** measures image size as a proxy for quality.
* **UnifiedReward** is a state-of-the-art reward model for multimodal understanding and generation, topping the human preference leaderboard.

        
## ✨ Important Hyperparameters
You can adjust the parameters in `config/grpo.py` to tune different hyperparameters. An empirical finding is that `config.sample.train_batch_size * num_gpu / config.sample.num_image_per_prompt * config.sample.num_batches_per_epoch = 48`, i.e., `group_number=48`, `group_size=24`.
Additionally, setting `config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2`.

## 🤗 Acknowledgement
This repo is based on [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) and [diffusers](https://github.com/huggingface/diffusers). We thank the authors for their valuable contributions to the AIGC community. Special thanks to Kevin Black for the excellent *ddpo-pytorch* repo.

## ⭐Citation
```
@article{liu2025flow,
  title={Flow-grpo: Training flow matching models via online rl},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Li, Yangguang and Liu, Jiaheng and Wang, Xintao and Wan, Pengfei and Zhang, Di and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2505.05470},
  year={2025}
}
```