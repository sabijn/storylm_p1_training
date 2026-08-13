import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F

import yaml
import argparse
from pathlib import Path
import wandb

from logger import setup_logging
from utils import load_tokenizer
from model import GPTLanguageModel
from prepare_data import create_dataloader, prepare_datasets

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    out = {}
    model.eval()

    for split, loader in {"train": train_loader, "val": val_loader}.items():
        losses = []
        loader_iter = iter(loader)

        for _ in range(eval_iters):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            batch = batch.to(device, non_blocking=True)
            x = batch[:, :-1]
            y = batch[:, 1:]

            _, loss = model(x, y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=Path, default='config.yaml')
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(config['training']['seed'])
    logger = setup_logging(config['paths']['logging_path'])

    run = wandb.init(
        entity= config['wandb']['entity'],
        project= config['wandb']['project'],
        dir=config['paths']['logging_path'],
        config={
            "learning_rate": config['training']['learning_rate'],
            "architecture": config['wandb']['architecture'],
            "dataset": config['wandb']['dataset'],
            "epochs": config['training']['max_iters'],
        },
    )

    # set up save names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path(config['paths']['output_model_base']) / f'{config['wandb']['architecture']}_{config['wandb']['dataset'].replace('+', '_')}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    EOS = config["tokens"]["eos_token"]
    BOS = config["tokens"]["bos_token"]

    special_tokens = {
        BOS: 100264,
        EOS: 100257,
    }

    tokenizer = load_tokenizer(special_tokens)

    logger.info("Loading and preparing datasets...")
    train_ds, val_ds = prepare_datasets(tokenizer, config, BOS, EOS)

    train_loader = create_dataloader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
    )

    val_loader = create_dataloader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
    )
    logger.info("Finished dataset preparation.")

    logger.info("Initializing model...")
    model = GPTLanguageModel(tokenizer.n_vocab, **config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"])
    )
    logger.info("Finished model initialization.")

    logger.info("Start training...")
    train_iter = iter(train_loader)

    for step in tqdm(range(config["training"]["max_iters"]), position=0):
        if step % config["training"]["eval_interval"] == 0:
            losses = estimate_loss(
                model,
                train_loader,
                val_loader,
                config["training"]["eval_iters"],
                device,
            )
            print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            run.log({"train_loss": losses['train'], "val_loss": losses['val']})
            torch.save(model.state_dict(), run_dir / f'{config['wandb']['architecture']}_{config['wandb']['dataset'].replace('+', '_')}_{step}.pt')

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device, non_blocking=True)
        xb = batch[:, :-1]
        yb = batch[:, 1:]

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    logger.info("Finished training.")

    logger.info("Saving model...")
    torch.save(model.state_dict(), run_dir / f'{config['wandb']['architecture']}_{config['wandb']['dataset'].replace('+', '_')}_{config['training']['max_iters']}.pt')
    wandb.save('model.h5')
    run.finish()