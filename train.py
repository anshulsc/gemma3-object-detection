import logging
import wandb
from functools import partial
from dataclasses import asdict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import Configuration
from utils import train_collate_function

import albumentations as A

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


augmentations = A.Compose([
    A.Resize(height=896, width=896),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], filter_invalid_bboxes=True))


def get_dataloader(processor, cfg):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=cfg.dtype, transform=augmentations
    )

    logger.info("Building data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
    )
    return train_dataloader


def train_model(model, optimizer, cfg, train_dataloader):
    logger.info("Start training")
    global_step = 0
    for epoch in range(cfg.epochs):
        for idx, batch in enumerate(train_dataloader):
            outputs = model(**batch.to(model.device))
            loss = outputs.loss
            if idx % 100 == 0:
                logger.info(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "epoch": epoch}, step=global_step)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
    return model


if __name__ == "__main__":
    cfg = Configuration()
   
    
    
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    train_dataloader = get_dataloader(processor, cfg)
    
    
    model_kwargs = {
        "torch_dtype": cfg.dtype,
        "device_map": cfg.device,
        "attn_implementation": "eager",
    }
    
    quantization_config = None
    if cfg.training_type == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["quantization_config"] = None
        
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        **model_kwargs,
    )
    
    if cfg.training_type == "full":
        logger.info("Getting model & turning only attention parameters to trainable for full fine-tuning")
        for name, param in model.named_parameters():
            if "attn" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    elif cfg.training_type in ["qlora", "lora"]:
        logger.info(f"Getting model & turning only attention parameters to trainable for {cfg.training_type.upper()}")
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        
        if cfg.training_type == "qlora":
            model = prepare_model_for_kbit_training(model, usq_gradient_checkpointing=True)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        

    model.train()
    model.to(cfg.device)

    # Credits to Sayak Paul for this beautiful expression
    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)
    
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in params_to_train)}")
    

    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name if hasattr(cfg, "run_name") else None,
        config=vars(cfg),
    )

    train_model(model, optimizer, cfg, train_dataloader)

    # Push the checkpoint to hub
    model.push_to_hub(cfg.checkpoint_id)
    processor.push_to_hub(cfg.checkpoint_id)

    wandb.finish()
    logger.info("Train finished")
