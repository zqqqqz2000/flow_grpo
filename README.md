# Flow-GRPO

This is an official implementation of Flow-GRPO: Training Flow Matching Models via Online RL.

## 🔔 News

[Update] We now provide an online demo for all three tasks at https://88750a65815db1a991.gradio.live/. You're welcome to try it out!

**[Update]** We release a new **GenEval model** that maintains image quality close to the **base model**, while still achieving the original **GenEval score of 95**. _Feel free to give it a try!_



## ✅ TODO
- [x] Provide a gradio demo
- [ ] Provide a **web demo** showcasing a wide range of generation examples for **GenEval**, **OCR**, and **PickScore**. _@GongyeLiu is working on this urgently._
- [ ] Provide a **web visualization** of image evolution during training for all three tasks. _@GongyeLiu is working on this urgently._

## Model
| Task    | Model |
| -------- | -------- |
| GenEval     | [🤗GenEval](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-GenEval) |
| Text Rendering     | [🤗Text](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-Text) |
| Human Preference Alignment     | [🤗PickScore](https://huggingface.co/jieliu/SD3.5M-FlowGRPO-PickScore) |

## Installation
```bash
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo
conda create -n flow_grpo python=3.10.16
pip install -e .
```

## Reward
The steps above only install the current repository. However, RL training requires different rewards, and each reward model might depend on some older pre-trained models. It's difficult to place all of these into a single Conda environment without version conflicts. Therefore, drawing inspiration from the ddpo-pytorch implementation, we use a remote server setup for some rewards.

### OCR
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

### GenEval
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server).

## Usage
Single-node training:
```bash
bash scripts/single_node/main.sh
```
Multi-node training:
```bash
# Master node
bash scripts/multi_node/main.sh
# Other nodes
bash scripts/multi_node/main1.sh
bash scripts/multi_node/main2.sh
```
### Multi Reward Training
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

        
## Important Hyperparameters
You can adjust the parameters in `config/dgx.py` to tune different hyperparameters. An empirical finding is that `config.sample.train_batch_size * num_gpu / config.sample.num_image_per_prompt * config.sample.num_batches_per_epoch = 48`, i.e., `group_number=48`, `group_size=24`.
Additionally, setting `config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2` also yields good performance.

## Acknowledgement
This repo is based on [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) and [diffusers](https://github.com/huggingface/diffusers). We thank the authors for their valuable contributions to the AIGC community. Special thanks to Kevin Black for the excellent *ddpo-pytorch* repo.

## Citation
```
@misc{liu2025flowgrpo,
      title={Flow-GRPO: Training Flow Matching Models via Online RL}, 
      author={Jie Liu and Gongye Liu and Jiajun Liang and Yangguang Li and Jiaheng Liu and Xintao Wang and Pengfei Wan and Di Zhang and Wanli Ouyang},
      year={2025},
      eprint={2505.05470},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.05470}, 
}
```