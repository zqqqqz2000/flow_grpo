# SDXL支持和简化奖励函数使用指南

本项目已更新以支持Stable Diffusion XL (SDXL)模型，并简化了奖励函数以仅使用unified reward。

## 主要更改

### 1. 新增的文件

- `flow_grpo/diffusers_patch/sdxl_pipeline_with_logprob.py` - SDXL pipeline，支持logprob计算
- `flow_grpo/diffusers_patch/sdxl_ddim_with_logprob.py` - SDXL的DDIM调度器，支持logprob
- `config/sdxl.py` - SDXL模型的配置文件
- `SDXL_USAGE.md` - 本使用指南

### 2. 修改的文件

- `scripts/train_sd3.py` - 更新为同时支持SD3和SDXL模型
- 简化了奖励函数，仅使用unified reward

## 使用方法

### 1. 训练SDXL模型

使用SDXL模型进行训练：

```bash
python scripts/train_sd3.py --config=config/sdxl.py:sdxl_base
```

### 2. 配置说明

在配置文件中，脚本会自动检测是否为SDXL模型：

```python
# 自动检测SDXL模型
is_sdxl = "stable-diffusion-xl" in config.pretrained.model.lower() or "sdxl" in config.pretrained.model.lower()
```

支持的SDXL模型：
- `stabilityai/stable-diffusion-xl-base-1.0`
- 任何包含"stable-diffusion-xl"或"sdxl"的模型名称

### 3. 关键差异

#### SDXL vs SD3 的主要区别：

1. **模型架构**：
   - SDXL使用UNet而不是Transformer
   - SDXL有两个文本编码器，而不是三个

2. **调度器**：
   - SDXL使用DDIM调度器
   - SD3使用Flow Matching调度器

3. **分辨率**：
   - SDXL最佳分辨率为1024x1024
   - SD3可以使用较低分辨率

4. **LoRA配置**：
   - SDXL和SD3有不同的target_modules

### 4. 简化的奖励函数

现在只使用unified reward函数，而不是多个奖励的组合：

```python
# 旧版本（多奖励）
config.reward_fn = {
    "aesthetic": 0.1,
    "pickscore": 0.5,
    "imagereward": 0.4
}

# 新版本（简化）
reward_fn = unifiedreward_score_sglang(device)
```

## 配置示例

### SDXL基础配置

```python
def sdxl_base():
    config = base.get_config()
    
    # SDXL模型
    config.pretrained.model = "stabilityai/stable-diffusion-xl-base-1.0"
    config.resolution = 1024  # SDXL最佳分辨率
    
    # 采样配置
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 8
    config.sample.guidance_scale = 7.5
    
    # 训练配置
    config.train.learning_rate = 1e-4
    config.train.beta = 0.01
    
    return config
```

## 注意事项

1. **内存需求**：SDXL模型比SD3模型更大，需要更多GPU内存
2. **分辨率**：建议使用1024x1024分辨率以获得最佳结果
3. **批次大小**：可能需要根据GPU内存调整批次大小
4. **奖励函数**：确保unified reward服务正在运行

## 故障排除

1. **内存不足**：减少batch_size或num_image_per_prompt
2. **导入错误**：确保安装了正确版本的diffusers库
3. **奖励计算错误**：检查unified reward服务是否正常运行

## 性能优化建议

1. 使用LoRA训练以减少内存使用
2. 开启EMA以提高训练稳定性
3. 调整guidance_scale以平衡质量和多样性
4. 使用混合精度训练(fp16)以提高速度 