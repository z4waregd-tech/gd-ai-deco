import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class GDTokenizer:
    def __init__(self):
        # We start with some special tokens
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[THEME]": 2,
            "[GP_START]": 3,
            "[GP_END]": 4,
            "[DECO_START]": 5,
            "[DECO_END]": 6,
            "<OBJ>": 7,
            "</OBJ>": 8
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = len(self.vocab)

    def get_id(self, token_str):
        if token_str not in self.vocab:
            self.vocab[token_str] = self.next_id
            self.inverse_vocab[self.next_id] = token_str
            self.next_id += 1
        return self.vocab[token_str]

    def tokenize_object(self, obj, is_gameplay=False):
        # Discretize X and Y to the nearest 15 units (half a GD block)
        x_raw = float(obj.get("2", 0))
        y_raw = float(obj.get("3", 0))
        x_snap = round(x_raw / 15) * 15
        y_snap = round(y_raw / 15) * 15

        obj_id = obj.get("1", "1")
        color = obj.get("21", "1") # 21 is color channel
        
        # New properties!
        rot_raw = float(obj.get("6", 0))
        rot_snap = int(round(rot_raw)) # Nearest integer degree
        
        scale_raw = float(obj.get("32", 1.0))
        scale_snap = round(scale_raw, 1) # 1 decimal place

        groups_str = obj.get("57", "")
        groups = [g for g in groups_str.split(".") if g]
        
        tokens = [
            "<OBJ>",
            f"<ID:{obj_id}>",
            f"<X:{x_snap}>",
            f"<Y:{y_snap}>"
        ]
        
        # Add properties if they are not default to save space
        if rot_snap != 0:
            tokens.append(f"<R:{rot_snap}>")
            
        if scale_snap != 1.0:
            tokens.append(f"<S:{scale_snap}>")
            
        for g in groups:
            tokens.append(f"<G:{g}>")
        
        if not is_gameplay:
            tokens.append(f"<C:{color}>")
            
        tokens.append("</OBJ>")
        return tokens

    def save(self, path="vocab.json"):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    def load(self, path="vocab.json"):
        with open(path, "r") as f:
            self.vocab = json.load(f)
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.next_id = max(self.vocab.values()) + 1


class GDLevelDataset(Dataset):
    def __init__(self, data_dir="datasets", chunk_size=900, tokenizer=None, max_length=512):
        self.tokenizer = tokenizer or GDTokenizer()
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.chunks = []

        data_path = Path(data_dir)
        
        for file in data_path.glob("*.json"):
            with open(file, "r") as f:
                level_data = json.load(f)
            self._process_level(level_data)
            
    def _process_level(self, level_data):
        theme = level_data.get("theme", "Unknown")
        gameplay = level_data.get("gameplay", [])
        deco = level_data.get("deco", [])
        channels = level_data.get("channels", {})
        
        # Find max X to know how many chunks we need
        max_x = 0
        for obj in gameplay + deco:
            x = float(obj.get("2", 0))
            if x > max_x:
                max_x = x
                
        num_chunks = int(max_x / self.chunk_size) + 1
        
        # Group objects by chunk
        chunked_gp = {i: [] for i in range(num_chunks)}
        chunked_deco = {i: [] for i in range(num_chunks)}
        
        for obj in gameplay:
            x = float(obj.get("2", 0))
            chunk_idx = int(x / self.chunk_size)
            # Normalize X relative to the chunk start
            obj["2"] = str(x - (chunk_idx * self.chunk_size))
            chunked_gp[chunk_idx].append(obj)
            
        for obj in deco:
            x = float(obj.get("2", 0))
            chunk_idx = int(x / self.chunk_size)
            obj["2"] = str(x - (chunk_idx * self.chunk_size))
            chunked_deco[chunk_idx].append(obj)
            
        for i in range(num_chunks):
            # Only keep chunks that actually have deco
            if not chunked_deco[i]:
                continue
                
            chunk_tokens = []
            
            # 1. Theme
            chunk_tokens.append("[THEME]")
            for theme_word in theme.replace(",", "").split():
                chunk_tokens.append(f"<T:{theme_word}>")
            
            # 2. Color Channels (We only feed channels to the first chunk (X=0) so it learns to initialize level colors)
            if i == 0 and channels:
                chunk_tokens.append("[CHANNELS_START]")
                for ch_id, rgb in channels.items():
                    chunk_tokens.extend([
                        f"<CH:{ch_id}>",
                        f"<CR:{rgb['r']}>",
                        f"<CG:{rgb['g']}>",
                        f"<CB:{rgb['b']}>"
                    ])
                chunk_tokens.append("[CHANNELS_END]")
            
            # 3. Gameplay
            chunk_tokens.append("[GP_START]")
            for obj in chunked_gp[i]:
                chunk_tokens.extend(self.tokenizer.tokenize_object(obj, is_gameplay=True))
            chunk_tokens.append("[GP_END]")
            
            # 4. Deco
            chunk_tokens.append("[DECO_START]")
            for obj in chunked_deco[i]:
                chunk_tokens.extend(self.tokenizer.tokenize_object(obj, is_gameplay=False))
            chunk_tokens.append("[DECO_END]")
            
            # Convert to IDs
            chunk_ids = [self.tokenizer.get_id(t) for t in chunk_tokens]
            self.chunks.append(chunk_ids)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Truncate or Pad to max_length
        if len(chunk) > self.max_length:
            chunk = chunk[:self.max_length]
            chunk[-1] = self.tokenizer.get_id("[DECO_END]") # ensure it ends properly
            
        pad_len = self.max_length - len(chunk)
        chunk = chunk + [self.tokenizer.get_id("[PAD]")] * pad_len
        
        # For training next-token prediction, x is input, y is the target (shifted by 1)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

if __name__ == "__main__":
    print("Building tokenized dataset...")
    tokenizer = GDTokenizer()
    # 512 is max tokens per chunk. If chunks are heavy on deco, we might need 1024 or 2048.
    dataset = GDLevelDataset(tokenizer=tokenizer, max_length=1024)
    
    # Save the vocabulary so our AI can understand the tokens later!
    tokenizer.save("vocab.json")
    
    print(f"Dataset created with {len(dataset)} chunks!")
    print(f"Vocabulary size: {len(tokenizer.vocab)} unique tokens")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for x, y in loader:
        print(f"Input batch shape: {x.shape}")
        print(f"Target batch shape: {y.shape}")
        break
