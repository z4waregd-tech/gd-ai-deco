import os
import torch
import torch.nn.functional as F
from dataset import GDTokenizer
from model import GDEncoderDecoderTransformer
import xml.etree.ElementTree as ET
import base64
import gzip
import collections

def generate_deco(theme="Hellish, Red, Demon", max_tokens=1024, chunk_size=150):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Generating on: {device}")

    tokenizer = GDTokenizer()
    try:
        tokenizer.load("vocab.json")
    except FileNotFoundError:
        print("Error: vocab.json not found! Train the model first.")
        return

    vocab_size = len(tokenizer.vocab)
    model = GDEncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=384,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1536,
        max_seq_len=1024,
    ).to(device)

    # Load checkpoint
    best_path = "gd_decorator_model_best.pth"
    latest_path = "gd_decorator_model.pth"
    ckpt = best_path if os.path.exists(best_path) else latest_path
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        print(f"Loaded checkpoint: {ckpt}")
    except FileNotFoundError:
        print("Error: No model checkpoint found! Run train.py first.")
        return

    try:
        from build_datasets import extract_level_data, decode_level_string, parse_level_objects
    except ImportError:
        print("Required functions from build_datasets.py not found.")
        return

    try:
        level_name, b64_data = extract_level_data("gameplay.gmd")
        raw_level_string = decode_level_string(b64_data)
        gameplay_objects = parse_level_objects(raw_level_string)
    except Exception as e:
        print(f"Failed to read gameplay.gmd: {e}")
        return

    # 1. Group gameplay objects into chunks (just like training!)
    chunked_gp = collections.defaultdict(list)
    max_x = 0
    for obj in gameplay_objects:
        x_raw = float(obj.get("2", 0))
        if x_raw > max_x:
            max_x = x_raw
        chunk_idx = int(x_raw / chunk_size) if x_raw >= 0 else int(x_raw // chunk_size)
        
        # We must clone obj so we don't mess up the original gameplay objects
        o = dict(obj)
        o["2"] = str(x_raw - (chunk_idx * chunk_size))
        chunked_gp[chunk_idx].append(o)

    # Sort the gameplay correctly from left to right as well!
    for i in chunked_gp:
        chunked_gp[i].sort(key=lambda o: (float(o.get("2", 0)), float(o.get("3", 0))))

    min_chunk = min(chunked_gp.keys()) if chunked_gp else 0
    num_chunks = int(max_x / chunk_size) + 1

    theme_tokens = ["[THEME]"]
    for word in theme.replace(",", "").split():
        theme_tokens.append(f"<T:{word}>")

    deco_start_id = tokenizer.vocab.get("[DECO_START]", 1)
    deco_end_id   = tokenizer.vocab.get("[DECO_END]",   1)

    print(f"\nDivided level into {num_chunks} chunks. Generating decoration...")
    global_generated_tokens = []
    
    # 2. Process Chunk by Chunk
    for i in range(min_chunk, num_chunks):
        print(f"\n--- Decorating Chunk {i} ---")
        src_tokens = list(theme_tokens)
        src_tokens.append("[GP_START]")
        
        for obj in chunked_gp.get(i, []):
            src_tokens.extend(tokenizer.tokenize_object(obj, is_gameplay=True))
            
        src_tokens.append("[GP_END]")
        
        max_src = 512
        src_ids = [tokenizer.vocab.get(t, tokenizer.vocab["[UNK]"]) for t in src_tokens]
        if len(src_ids) > max_src:
            print(f"  (Gameplay too dense, trimming to {max_src} tokens)")
            src_ids = src_ids[:max_src]

        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            memory = model.encode(src_tensor)   # [1, src_len, d_model]

        generated_ids = [deco_start_id]

        with torch.no_grad():
            for step in range(max_tokens):
                tgt_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
                logits = model.decode_step(tgt_tensor, memory)
                next_logits = logits[0, -1, :]

                # Smart Repetition Penalty (Short-term memory)
                # This breaks "infinite loops" (stacking at X=0) without preventing you from building walls.
                rep_penalty = 1.08
                if len(generated_ids) > 1:
                    # Only look at the last 30 tokens (about 4-5 objects)
                    recent = set(generated_ids[-30:])
                    for token_id in recent:
                        if 0 <= token_id < next_logits.shape[0]:
                            if next_logits[token_id] > 0:
                                next_logits[token_id] /= rep_penalty
                            else:
                                next_logits[token_id] *= rep_penalty

                # Increased temperature to 0.85 to allow for natural variety
                temperature = 0.85
                scaled = next_logits / temperature
                top_p = 0.95

                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                remove_full = remove.scatter(-1, sorted_idx, remove)
                scaled[remove_full] = -float("Inf")

                probs = F.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                if next_id == deco_end_id:
                    print(f"AI seamlessly finished chunk {i}!")
                    break

                generated_ids.append(next_id)

        # 3. Convert back to tokens and Restore Global X Coordinate
        chunk_tokens = []
        for id_val in generated_ids[1:]:
            token_str = tokenizer.inverse_vocab.get(id_val, "[UNK]")
            
            # If it's an X coordinate, add the chunk's offset back so it matches the real level length!
            if token_str.startswith("<X:"):
                local_x = int(token_str[3:-1])
                global_x = local_x + (i * chunk_size)
                token_str = f"<X:{global_x}>"
                
            chunk_tokens.append(token_str)
            
        global_generated_tokens.extend(chunk_tokens)

    # ── 4. Parse all chunks back into GD object format ───────────────────────
    print("\n\nPacking everything beautifully into ai_decorated.gmd...")
    gd_string = ""
    current_obj = {}
    channels_dict = {}
    current_ch = None

    for token in global_generated_tokens:
        if token.startswith("<CH:"):
            current_ch = token[4:-1]
            if current_ch not in channels_dict:
                channels_dict[current_ch] = {"r": "255", "g": "255", "b": "255"}
        elif token.startswith("<CR:") and current_ch:
            channels_dict[current_ch]["r"] = token[4:-1]
        elif token.startswith("<CG:") and current_ch:
            channels_dict[current_ch]["g"] = token[4:-1]
        elif token.startswith("<CB:") and current_ch:
            channels_dict[current_ch]["b"] = token[4:-1]
        elif token == "<OBJ>":
            current_obj = {}
        elif token.startswith("<ID:"):
            current_obj["1"] = token[4:-1]
        elif token.startswith("<X:"):
            current_obj["2"] = token[3:-1]
        elif token.startswith("<Y:"):
            current_obj["3"] = token[3:-1]
        elif token.startswith("<C:"):
            current_obj["21"] = token[3:-1]
        elif token.startswith("<ZL:"):
            current_obj["25"] = token[4:-1]
        elif token.startswith("<ZO:"):
            current_obj["24"] = token[4:-1]
        elif token.startswith("<R:"):
            current_obj["6"] = token[3:-1]
        elif token.startswith("<S:"):
            current_obj["32"] = token[3:-1]
        elif token.startswith("<G:"):
            g_id = token[3:-1]
            current_obj["57"] = (current_obj["57"] + f".{g_id}") if "57" in current_obj else g_id
        elif token == "</OBJ>":
            if "1" in current_obj:
                gd_string += ",".join(f"{k},{v}" for k, v in current_obj.items()) + ";"

    # ── 5. Inject color channels & rebuild level string ───────────────────────
    ch_string = ""
    for ch_id, rgb in channels_dict.items():
        ch_string += f"1_{rgb['r']}_2_{rgb['g']}_3_{rgb['b']}_6_{ch_id}|"

    header_parts = raw_level_string.split(";", 1)
    if ch_string:
        if "kS38" in header_parts[0]:
            header_parts[0] = header_parts[0].replace("kS38,", f"kS38,{ch_string}")
        else:
            header_parts[0] += f",kS38,{ch_string}"

    final_level_string = header_parts[0] + ";" + header_parts[1] + gd_string

    compressed = gzip.compress(final_level_string.encode("utf-8"))
    b64_out = base64.urlsafe_b64encode(compressed).decode("utf-8")

    tree = ET.parse("gameplay.gmd")
    root = tree.getroot()
    children = list(root.find("dict"))
    for j in range(len(children) - 1):
        if children[j].tag == "k" and children[j].text == "k4":
            children[j + 1].text = b64_out

    tree.write("ai_decorated.gmd", encoding="utf-8")
    print("\nSUCCESS: Saved fully playable level to ai_decorated.gmd")


if __name__ == "__main__":
    generate_deco(theme="Hellish, Red, Demon")
