import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class GDTokenizer:
    def __init__(self):
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
        # Discretize X and Y to the nearest 2 units (much more precise than 15)
        x_raw = float(obj.get("2", 0))
        y_raw = float(obj.get("3", 0))
        x_snap = int(round(x_raw / 2) * 2)
        y_snap = int(round(y_raw / 2) * 2)

        obj_id = obj.get("1", "1")
        color = obj.get("21", "1") # 21 is color channel
        
        # Layering properties
        z_layer = obj.get("25", "0") # Z Layer (B4, B3, B2, B1, T1, T2, T3)
        z_order = obj.get("24", "0") # Z Order (priority within layer)
        
        # Transform properties
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

        # Add layering if not default
        if z_layer != "0":
            tokens.append(f"<ZL:{z_layer}>")
        if z_order != "0":
            tokens.append(f"<ZO:{z_order}>")
        
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
        
        # Find max X and min X to know how many chunks we need
        min_x = 0
        max_x = 0
        for obj in gameplay + deco:
            x = float(obj.get("2", 0))
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
                
        min_chunk = int(min_x / self.chunk_size)
        if min_x < 0 and min_x % self.chunk_size != 0:
            min_chunk -= 1
            
        num_chunks = int(max_x / self.chunk_size) + 1
        
        # Group objects by chunk
        import collections
        chunked_gp = collections.defaultdict(list)
        chunked_deco = collections.defaultdict(list)
        
        for obj in gameplay:
            x = float(obj.get("2", 0))
            chunk_idx = int(x / self.chunk_size) if x >= 0 else int(x // self.chunk_size)
            # Normalize X relative to the chunk start
            obj["2"] = str(x - (chunk_idx * self.chunk_size))
            chunked_gp[chunk_idx].append(obj)
            
        for obj in deco:
            x = float(obj.get("2", 0))
            chunk_idx = int(x / self.chunk_size) if x >= 0 else int(x // self.chunk_size)
            obj["2"] = str(x - (chunk_idx * self.chunk_size))
            chunked_deco[chunk_idx].append(obj)
            
        def sort_objects(obj_list):
            obj_list.sort(key=lambda o: (float(o.get("2", 0)), float(o.get("3", 0))))

        for i in chunked_gp:
            sort_objects(chunked_gp[i])
        for i in chunked_deco:
            sort_objects(chunked_deco[i])
            
        for i in range(min_chunk, num_chunks):
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

class GDEncoderDecoderDataset(Dataset):
    """
    Returns separate encoder (gameplay) and decoder (deco) sequences.
    Each sample is a (src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask) tuple:

      src      — [THEME] ... [GP_START] <gameplay objects> [GP_END]
      tgt_in   — [DECO_START] <deco objects>          (teacher forcing input)
      tgt_out  — <deco objects> [DECO_END]             (prediction targets)
      *_pad_mask — True where the token is padding (for PyTorch Transformer)
    """

    def __init__(self, data_dir="datasets", chunk_size=900, tokenizer=None,
                 max_src_len=512, max_tgt_len=512):
        self.tokenizer = tokenizer or GDTokenizer()
        self.chunk_size = chunk_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.samples = []  # list of (src_tokens, tgt_tokens)

        data_path = Path(data_dir)
        for file in data_path.glob("*.json"):
            with open(file, "r") as f:
                level_data = json.load(f)
            self._process_level(level_data)

        # !! CRITICAL: Force-register every token NOW so that vocab_size in train.py
        # is computed AFTER all IDs exist. Without this, __getitem__ adds new IDs
        # at training time causing embedding index-out-of-bounds on CUDA.
        self._build_vocab()

    def _build_vocab(self):
        """Pre-warm the tokenizer vocab with every token that will appear in training."""
        for src_tok, tgt_in_tok, tgt_out_tok in self.samples:
            for t in src_tok:
                self.tokenizer.get_id(t)
            for t in tgt_in_tok:
                self.tokenizer.get_id(t)
            for t in tgt_out_tok:
                self.tokenizer.get_id(t)


    def _process_level(self, level_data):
        import collections
        theme = level_data.get("theme", "Unknown")
        gameplay = level_data.get("gameplay", [])
        deco = level_data.get("deco", [])
        channels = level_data.get("channels", {})

        min_x, max_x = 0.0, 0.0
        for obj in gameplay + deco:
            x = float(obj.get("2", 0))
            min_x = min(min_x, x)
            max_x = max(max_x, x)

        min_chunk = int(min_x / self.chunk_size)
        if min_x < 0 and min_x % self.chunk_size != 0:
            min_chunk -= 1
        num_chunks = int(max_x / self.chunk_size) + 1

        chunked_gp = collections.defaultdict(list)
        chunked_deco = collections.defaultdict(list)

        for obj in gameplay:
            x = float(obj.get("2", 0))
            ci = int(x / self.chunk_size) if x >= 0 else int(x // self.chunk_size)
            o = dict(obj); o["2"] = str(x - ci * self.chunk_size)
            chunked_gp[ci].append(o)

        for obj in deco:
            x = float(obj.get("2", 0))
            ci = int(x / self.chunk_size) if x >= 0 else int(x // self.chunk_size)
            o = dict(obj); o["2"] = str(x - ci * self.chunk_size)
            chunked_deco[ci].append(o)

        # CRITICAL FIX: Sort objects precisely by X then Y 
        # This completely stops the "random jumping" geometry dash does in save files.
        def sort_objects(obj_list):
            obj_list.sort(key=lambda o: (float(o.get("2", 0)), float(o.get("3", 0))))

        for i in chunked_gp:
            sort_objects(chunked_gp[i])
        for i in chunked_deco:
            sort_objects(chunked_deco[i])

        theme_tokens = ["[THEME]"]
        for w in theme.replace(",", "").split():
            theme_tokens.append(f"<T:{w}>")

        for i in range(min_chunk, num_chunks):
            if not chunked_deco[i]:
                continue

            # ── Encoder sequence (gameplay context) ──────────────────────────
            src_tokens = list(theme_tokens)
            if i == 0 and channels:
                src_tokens.append("[CHANNELS_START]")
                for ch_id, rgb in channels.items():
                    src_tokens += [
                        f"<CH:{ch_id}>", f"<CR:{rgb['r']}>",
                        f"<CG:{rgb['g']}>", f"<CB:{rgb['b']}>"
                    ]
                src_tokens.append("[CHANNELS_END]")

            src_tokens.append("[GP_START]")
            for obj in chunked_gp[i]:
                src_tokens.extend(self.tokenizer.tokenize_object(obj, is_gameplay=True))
            src_tokens.append("[GP_END]")

            # ── Decoder sequence (decoration) ────────────────────────────────
            deco_tokens = []
            for obj in chunked_deco[i]:
                deco_tokens.extend(self.tokenizer.tokenize_object(obj, is_gameplay=False))

            tgt_in_tokens = ["[DECO_START]"] + deco_tokens
            tgt_out_tokens = deco_tokens + ["[DECO_END]"]

            self.samples.append((src_tokens, tgt_in_tokens, tgt_out_tokens))

    @staticmethod
    def _pad(ids, max_len, pad_id):
        ids = ids[:max_len]
        mask = [False] * len(ids) + [True] * (max_len - len(ids))
        ids = ids + [pad_id] * (max_len - len(ids))
        return ids, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_tok, tgt_in_tok, tgt_out_tok = self.samples[idx]
        pad_id = self.tokenizer.get_id("[PAD]")

        src_ids = [self.tokenizer.get_id(t) for t in src_tok]
        tgt_in_ids = [self.tokenizer.get_id(t) for t in tgt_in_tok]
        tgt_out_ids = [self.tokenizer.get_id(t) for t in tgt_out_tok]

        src_ids, src_mask = self._pad(src_ids, self.max_src_len, pad_id)
        tgt_in_ids, tgt_mask = self._pad(tgt_in_ids, self.max_tgt_len, pad_id)
        tgt_out_ids, _ = self._pad(tgt_out_ids, self.max_tgt_len, pad_id)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_in_ids, dtype=torch.long),
            torch.tensor(tgt_out_ids, dtype=torch.long),
            torch.tensor(src_mask, dtype=torch.bool),
            torch.tensor(tgt_mask, dtype=torch.bool),
        )



if __name__ == "__main__":
    print("Building tokenized dataset...")
    tokenizer = GDTokenizer()
    # 1024 is max tokens per chunk. 
    dataset = GDLevelDataset(tokenizer=tokenizer, max_length=1024)
    
    # Save the vocabulary so our AI can understand the tokens later!
    tokenizer.save("vocab.json")
    
    print(f"Dataset created with {len(dataset)} chunks!")
    print(f"Vocabulary size: {len(tokenizer.vocab)} unique tokens")
    
    # Test loader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for x, y in loader:
        print(f"Input batch shape: {x.shape}")
        print(f"Target batch shape: {y.shape}")
        break
