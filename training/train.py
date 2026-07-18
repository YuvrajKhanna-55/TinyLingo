import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "tinylingo"))
sys.path.append(os.path.join(PROJECT_ROOT, "tokeniser"))

import time
import torch
import torch.nn as nn
from tokenizer_wrapper import TranslationTokenizer
from transformer import Transformer
from dataset import make_dataloader


CONFIG = {
    "d_model": 256,
    "num_heads": 8,
    "d_ff": 1024,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "max_seq_len": 64,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "grad_clip_norm": 1.0,
    "log_every_n_batches": 100,
    "save_every_n_steps": 2000,   
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "training", "checkpoints"),
    "resume_from": None,
}

def save_checkpoint(path, model, optimizer, epoch, global_step, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": CONFIG,
    }, path)
    print(f"    [checkpoint saved: {path}]")

def get_lr_schedule(d_model, warmup_steps=4000):
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return lr_lambda

def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Resumed from {path} -- epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    return checkpoint["epoch"], checkpoint["global_step"], checkpoint["best_val_loss"]


def run_epoch(model, dataloader, optimizer, loss_fn, pad_id, device,
              training, epoch_num, global_step, log_every=100, on_step_end=None):
    model.train() if training else model.eval()

    total_loss, total_examples = 0.0, 0
    start_time = time.time()

    with torch.set_grad_enabled(training):
        for batch_idx, (src_tensor, tgt_tensor) in enumerate(dataloader):
            src_tensor = src_tensor.to(device)
            tgt_tensor = tgt_tensor.to(device)

            decoder_input = tgt_tensor[:, :-1]
            decoder_label = tgt_tensor[:, 1:]

            if training:
                optimizer.zero_grad()

            logits = model(src_tensor, decoder_input, pad_id)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), decoder_label.reshape(-1))

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if on_step_end is not None:
                    on_step_end(global_step)

            batch_size = src_tensor.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            if training and (batch_idx + 1) % log_every == 0:
                elapsed = time.time() - start_time
                print(f"  epoch {epoch_num} | batch {batch_idx+1}/{len(dataloader)} "
                      f"| loss {loss.item():.4f} | {elapsed:.1f}s elapsed")

    avg_loss = total_loss / total_examples
    return avg_loss, global_step

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("config:", CONFIG)

    tok = TranslationTokenizer(
        os.path.join(PROJECT_ROOT, "tokeniser", "artifacts", "tinylingo_spm.model"),
        max_seq_len=CONFIG["max_seq_len"]
    )

    train_loader = make_dataloader(
        os.path.join(PROJECT_ROOT, "dataset", "processed", "tinylingo_train.parquet"),
        tok, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = make_dataloader(
        os.path.join(PROJECT_ROOT, "dataset", "processed", "tinylingo_val.parquet"),
        tok, batch_size=CONFIG["batch_size"], shuffle=False
    )

    model = Transformer(
        vocab_size=tok.vocab_size,
        d_model=CONFIG["d_model"], num_heads=CONFIG["num_heads"], d_ff=CONFIG["d_ff"],
        num_encoder_layers=CONFIG["num_encoder_layers"], num_decoder_layers=CONFIG["num_decoder_layers"],
        max_seq_len=CONFIG["max_seq_len"], dropout=CONFIG["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_schedule(CONFIG["d_model"]))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_epoch, global_step, best_val_loss = 0, 0, float("inf")

    if CONFIG["resume_from"] is not None:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            CONFIG["resume_from"], model, optimizer, device
        )
        start_epoch += 1

    latest_path = os.path.join(CONFIG["checkpoint_dir"], "latest.pt")
    best_path = os.path.join(CONFIG["checkpoint_dir"], "best.pt")

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        print(f"\n=== Epoch {epoch} ===")

        def mid_epoch_save(step, ep=epoch, bvl_ref=lambda: best_val_loss):
            if step % CONFIG["save_every_n_steps"] == 0:
                save_checkpoint(latest_path, model, optimizer, ep, step, bvl_ref())

        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, loss_fn, tok.pad_id, device,
            training=True, epoch_num=epoch, global_step=global_step,
            log_every=CONFIG["log_every_n_batches"], on_step_end=mid_epoch_save
        )
        print(f"Epoch {epoch} train loss: {train_loss:.4f}")

        val_loss, _ = run_epoch(
            model, val_loader, optimizer, loss_fn, tok.pad_id, device,
            training=False, epoch_num=epoch, global_step=global_step
        )
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")

        save_checkpoint(latest_path, model, optimizer, epoch, global_step, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, global_step, best_val_loss)
            print(f"  New best val loss: {best_val_loss:.4f}")