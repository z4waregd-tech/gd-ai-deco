import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GDTokenizer, GDLevelDataset
from model import GDAutoregressiveTransformer
import os
import sys

from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading tokenizer and dataset...")
    tokenizer = GDTokenizer()
    dataset = GDLevelDataset(tokenizer=tokenizer, max_length=1024) 
    
    # Save the vocab so our generation script can use it later
    tokenizer.save("vocab.json")
    
    # Dataloader loops over chunks in random order
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model (deeper: 8 layers for higher accuracy)
    vocab_size = len(tokenizer.vocab)
    print(f"Initializing model with vocab size {vocab_size}...")
    
    model = GDAutoregressiveTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=8,        # Upgraded: 4 -> 8 layers (more brain power!)
        dim_feedforward=1024,
        max_seq_len=1024
    ).to(device)
    
    # 3. Optimization Setup
    pad_idx = tokenizer.get_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Learning Rate Scheduler: Slow down as we get smarter for higher accuracy
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    
    # 4. Training Loop
    print("\nStarting Training! (Save gracefully via 'q' on Windows or Ctrl+C)")
    epoch = 0
    save_path = os.path.abspath("gd_decorator_model.pth")
    print(f"Model will be saved to: {save_path}")
    
    try:
        while True:
            model.train()
            total_loss = 0
            total_correct = 0
            total_tokens = 0
            
            progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")
            
            for batch_idx, (x, y) in progress_bar:
                if sys.platform == "win32":
                    import msvcrt
                    if msvcrt.kbhit():
                        if msvcrt.getch().decode('utf-8').lower() == 'q':
                            print("\nReceived 'q', stopping training gracefully...")
                            raise KeyboardInterrupt
                        
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                logits = model(x)
                logits_flat = logits.view(-1, vocab_size)
                y_flat = y.view(-1)
                
                loss = criterion(logits_flat, y_flat)
                loss.backward()
                
                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate Accuracy
                with torch.no_grad():
                    valid_mask = y_flat != pad_idx
                    predictions = torch.argmax(logits_flat, dim=-1)
                    correct = (predictions[valid_mask] == y_flat[valid_mask]).sum().item()
                    total_correct += correct
                    total_tokens += valid_mask.sum().item()
                    
                    current_acc = (correct / valid_mask.sum().item()) * 100 if valid_mask.sum().item() > 0 else 0
                    
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{current_acc:.2f}%", "LR": f"{scheduler.get_last_lr()[0]:.6f}"})
                
            avg_loss = total_loss / len(loader)
            avg_acc = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
            print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} | Average Accuracy: {avg_acc:.2f}%")
            
            # Step the LR Scheduler
            scheduler.step()
            
            # Auto-save after every epoch!
            torch.save(model.state_dict(), save_path)
            
            epoch += 1
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Proceeding to save...")

        
    print("\nTraining Complete! Saving model...")
    torch.save(model.state_dict(), save_path)
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    train()
