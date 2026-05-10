import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GDTokenizer, GDEncoderDecoderDataset
from model import GDEncoderDecoderTransformer
import os
import sys
from tqdm import tqdm


def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("Loading tokenizer and dataset...")
    tokenizer = GDTokenizer()
    # CRITICAL: chunk_size=150 (5 GD blocks) instead of 900. 
    # With max_tgt_len=1024, the AI will now see the entire chunk instead of getting truncated at X=0!
    dataset = GDEncoderDecoderDataset(tokenizer=tokenizer, chunk_size=150, max_src_len=512, max_tgt_len=1024)
    tokenizer.save("vocab.json")
    # Reduced batch size to 4 because max_tgt_len=1024 requires 4x more VRAM!
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── 2. Model ─────────────────────────────────────────────────────────────
    vocab_size = len(tokenizer.vocab)
    print(f"Initializing Encoder-Decoder model with vocab size {vocab_size}...")

    model = GDEncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=384,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1536,
        dropout=0.1,
        max_seq_len=1024,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # --- Resume Training Logic ---
    save_path = os.path.abspath("gd_decorator_model.pth")
    best_path = os.path.abspath("gd_decorator_model_best.pth")
    
    if os.path.exists(save_path):
        print(f"RESUMING: Found existing checkpoint at {save_path}!")
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    elif os.path.exists(best_path):
        print(f"RESUMING: Found best checkpoint at {best_path}!")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    else:
        print("No existing model found. Starting from scratch.")

    # ── 3. Optimisation ───────────────────────────────────────────────────────
    pad_idx = tokenizer.get_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    # OneCycleLR provides a smooth warmup and decay.
    # With a 24k vocab, we use a slightly higher max_lr (6e-4).
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=6e-4,
        steps_per_epoch=len(loader), epochs=500,
        pct_start=5.0 / 500.0,  # 5 epochs of warmup
    )

    # ── 4. Training Loop ──────────────────────────────────────────────────────
    print("\nStarting Encoder-Decoder Training! (Save via 'q' on Windows or Ctrl+C)")
    save_path = os.path.abspath("gd_decorator_model.pth")
    best_path = os.path.abspath("gd_decorator_model_best.pth")
    print(f"Model will be saved to: {save_path}")
    print(f"Best model tracked at:  {best_path}")

    epoch = 0
    best_acc = 0.0
    try:
        while True:
            model.train()
            total_loss = 0
            total_correct = 0
            total_tokens = 0

            bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}")
            for batch_idx, (src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask) in bar:
                # ── Graceful quit on Windows ──
                if sys.platform == "win32":
                    import msvcrt
                    if msvcrt.kbhit():
                        if msvcrt.getch().decode("utf-8", errors="ignore").lower() == "q":
                            print("\nReceived 'q', stopping training gracefully...")
                            raise KeyboardInterrupt

                src = src.to(device)
                tgt_in = tgt_in.to(device)
                tgt_out = tgt_out.to(device)
                src_pad_mask = src_pad_mask.to(device)
                tgt_pad_mask = tgt_pad_mask.to(device)

                optimizer.zero_grad()

                # Forward: encoder reads gameplay, decoder reads deco shifted-right
                logits = model(
                    src, tgt_in,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                )

                # Loss over decoder outputs
                logits_flat = logits.reshape(-1, vocab_size)
                tgt_flat = tgt_out.reshape(-1)
                loss = criterion(logits_flat, tgt_flat)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                with torch.no_grad():
                    valid_mask = tgt_flat != pad_idx
                    preds = torch.argmax(logits_flat, dim=-1)
                    correct = (preds[valid_mask] == tgt_flat[valid_mask]).sum().item()
                    total_correct += correct
                    total_tokens += valid_mask.sum().item()
                    batch_acc = correct / max(valid_mask.sum().item(), 1) * 100

                bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{batch_acc:.2f}%"})

            avg_loss = total_loss / len(loader)
            avg_acc = total_correct / max(total_tokens, 1) * 100
            print(f"Epoch {epoch + 1} Completed | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}%")

            # Always save latest
            torch.save(model.state_dict(), save_path)

            # Save best separately
            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save(model.state_dict(), best_path)
                print(f"  ★ New best! ({best_acc:.2f}%) saved to {best_path}")

            epoch += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving...")

    torch.save(model.state_dict(), save_path)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    train()
