import imp
import os

import ml_collections

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def sdxl_base():
    config = base.get_config()

    # Use SDXL base model
    config.pretrained.model = "stabilityai/stable-diffusion-xl-base-1.0"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.num_epochs = 100
    config.use_lora = True

    # SDXL works best with 1024x1024 resolution
    config.resolution = 1024

    # Sampling configuration
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 4
    config.sample.num_steps = 50
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 7.5

    # Training configuration
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 2
    config.train.learning_rate = 1e-4
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 1.0
    config.train.beta = 0.01
    config.train.ema = True

    # Use only unifiedreward (simplified)
    config.reward_fn = {"unifiedreward": 1.0}

    # Prompting
    config.prompt_fn = "general_ocr"

    # Per-prompt stat tracking
    config.per_prompt_stat_tracking = True

    # Save configuration
    config.save_freq = 20
    config.eval_freq = 20
    config.save_dir = "logs/sdxl/base"

    return config


def get_config(name):
    return globals()[name]()
