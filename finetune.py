"""
finetune.py — Fine-tune the GD Decorator on a specific theme/style.

This loads your existing best.pth and continues training with a very low
learning rate so it specialises on one style without forgetting everything.

Usage:
    python finetune.py --theme "Hellish, Red, Demon" --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GDTokenizer, GDEncoderDecoderDataset
from model import GDEncoderDecoderTransformer
import os
import argparse
from tqdm import tqdm


def finetune(theme: str = "Hellish, Red, Demon", epochs: int = 10):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Fine-tuning on: {device}")

    # ── 1. Load dataset (same as training) ────────────────────────────────────
    print("Loading tokenizer and dataset...")
    tokenizer = GDTokenizer()
    dataset = GDEncoderDecoderDataset(
        tokenizer=tokenizer, chunk_size=150, max_src_len=512, max_tgt_len=1024
    )

    gpu_count = torch.cuda.device_count()
    # Use a smaller batch for finetuning — more careful, more precise updates
    batch_size = max(4, 4 * gpu_count)
    print(f"GPUs detected: {gpu_count} | Batch size: {batch_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── 2. Load the existing checkpoint ───────────────────────────────────────
    best_path   = os.path.abspath("gd_decorator_model_best.pth")
    save_path   = os.path.abspath("gd_decorator_model_ft.pth")  # separate file!

    if not os.path.exists(best_path):
        print(f"ERROR: {best_path} not found. Train the base model first.")
        return

    print(f"Loading base checkpoint: {best_path}")
    raw_ckpt = torch.load(best_path, map_location=device, weights_only=False)

    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        state_dict = raw_ckpt["model"]
        tokenizer.vocab        = raw_ckpt["vocab"]
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.next_id      = max(tokenizer.vocab.values()) + 1
        print(f"  Vocab restored from checkpoint ({len(tokenizer.vocab)} tokens).")
    else:
        state_dict = raw_ckpt  # legacy format

    # Strip DataParallel 'module.' prefix
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    vocab_size = cleaned["embedding.weight"].shape[0]
    print(f"Model vocab size: {vocab_size}")

    model = GDEncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=384,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1536,
        dropout=0.05,           # Lower dropout during finetuning
        max_seq_len=1024,
    ).to(device)

    model.load_state_dict(cleaned)
    print("Base model loaded! Starting fine-tune...")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # ── 3. Freeze encoder layers (optional but helps with style specialisation) 
    # Comment out the block below if you want FULL fine-tuning instead.
    print("Freezing encoder layers (decoder stays trainable)...")
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable:,}")

    # ── 4. Optimisation ────────────────────────────────────────────────────────
    pad_idx   = tokenizer.get_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.05)

    # KEY: Very low LR — we don't want to destroy what it already knows!
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-3
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Load fine-tuned checkpoint if resuming
    if os.path.exists(save_path):
        print(f"RESUMING fine-tune from {save_path}")
        ft_ckpt = torch.load(save_path, map_location=device, weights_only=False)
        if isinstance(ft_ckpt, dict) and "model" in ft_ckpt:
            m = model.module if hasattr(model, "module") else model
            m.load_state_dict(ft_ckpt["model"])

    # ── 5. Training Loop ───────────────────────────────────────────────────────
    best_acc  = 0.0
    vocab_size_model = vocab_size  # use the model's actual vocab size

    for epoch in range(epochs):
        model.train()
        total_loss = total_correct = total_tokens = 0

        pbar = tqdm(loader, desc=f"Finetune Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
            src = src.to(device); tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(src, tgt_in,
                                   src_key_padding_mask=src_pad_mask,
                                   tgt_key_padding_mask=tgt_pad_mask)
                    logits_flat = logits.reshape(-1, vocab_size_model)
                    tgt_flat    = tgt_out.reshape(-1)
                    loss        = criterion(logits_flat, tgt_flat)

                if torch.isnan(loss):
                    print("\n  [WARN] NaN loss! Skipping batch.")
                    optimizer.zero_grad(); continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(src, tgt_in,
                               src_key_padding_mask=src_pad_mask,
                               tgt_key_padding_mask=tgt_pad_mask)
                logits_flat = logits.reshape(-1, vocab_size_model)
                tgt_flat    = tgt_out.reshape(-1)
                loss        = criterion(logits_flat, tgt_flat)

                if torch.isnan(loss):
                    print("\n  [WARN] NaN loss! Skipping batch.")
                    optimizer.zero_grad(); continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                valid_mask  = tgt_flat != pad_idx
                preds       = torch.argmax(logits_flat, dim=-1)
                correct     = (preds[valid_mask] == tgt_flat[valid_mask]).sum().item()
                total_correct += correct
                total_tokens  += valid_mask.sum().item()
                batch_acc     = correct / max(valid_mask.sum().item(), 1) * 100

            pbar.set_postfix(Loss=f"{loss.item():.4f}", Acc=f"{batch_acc:.2f}%")

        avg_loss = total_loss / len(loader)
        avg_acc  = total_correct / max(total_tokens, 1) * 100
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")

        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ckpt  = {"model": state, "vocab": tokenizer.vocab}
        torch.save(ckpt, save_path)
        print(f"  Saved fine-tuned checkpoint → {save_path}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(ckpt, save_path.replace(".pth", "_best.pth"))
            print(f"  ★ New best fine-tune! ({best_acc:.2f}%)")

    print(f"\n✅ Fine-tuning complete! Best acc: {best_acc:.2f}%")
    print(f"   Use 'gd_decorator_model_ft_best.pth' in generate.py for this style.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme",  type=str, default="Hellish, Red, Demon",
                        help="Theme/style for fine-tuning")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of fine-tuning epochs")
    args = parser.parse_args()
    finetune(theme=args.theme, epochs=args.epochs)
