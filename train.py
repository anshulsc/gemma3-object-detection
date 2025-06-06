import logging
import wandb
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from config import Configuration
from utils import train_collate_function
import argparse
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


def get_dataloader(processor, args, dtype):
    logger.info("Fetching the dataset")
    train_dataset = load_dataset(cfg.dataset_id, split="train")
    train_collate_fn = partial(
        train_collate_function, processor=processor, dtype=dtype, transform=augmentations
    )

    logger.info("Building data loader")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
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
    cfg = Configuration.from_args()

    # Get values dynamicaly from user
    parser = argparse.ArgumentParser(description="Training for PaLiGemma")
    parser.add_argument('--model_id', type=str, required=True, default=cfg.model_id, help='Enter Huggingface Model ID')
    parser.add_argument('--dataset_id', type=str, required=True ,default=cfg.dataset_id, help='Enter Huggingface Dataset ID')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, help='Enter Batch Size')
    parser.add_argument('--lr', type=float, default=cfg.learning_rate, help='Enter Learning Rate')
    parser.add_argument('--checkpoint_id', type=str, required=True, default=cfg.checkpoint_id, help='Enter Huggingface Repo ID to push model')

    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_id)
    train_dataloader = get_dataloader(processor=processor, args=args, dtype=cfg.dtype)

    logger.info("Getting model & turning only attention parameters to trainable")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        device_map="cpu",
        attn_implementation="eager",
    )
    for name, param in model.named_parameters():
        if "attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.train()
    model.to(cfg.device)

    # Credits to Sayak Paul for this beautiful expression
    params_to_train = list(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(params_to_train, lr=args.lr)

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
