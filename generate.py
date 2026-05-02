import torch
import torch.nn.functional as F
from dataset import GDTokenizer
from model import GDAutoregressiveTransformer
import xml.etree.ElementTree as ET
import base64
import gzip

def generate_deco(theme="Hellish, Red, Demon, 2.1.", max_tokens=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Generating on: {device}")
    
    # 1. Load Tokenizer & Model
    tokenizer = GDTokenizer()
    try:
        tokenizer.load("vocab.json")
    except FileNotFoundError:
        print("Error: vocab.json not found! You must train the model first.")
        return
        
    vocab_size = len(tokenizer.vocab)
    model = GDAutoregressiveTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=1024,
        max_seq_len=1024
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("gd_decorator_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Error: gd_decorator_model.pth not found! Run train.py first.")
        return

    print("Model loaded successfully!")
    
    # 2. Extract real gameplay from gameplay.gmd
    try:
        from build_datasets import extract_level_data, decode_level_string, parse_level_objects
    except ImportError:
        print("Required functions from build_datasets.py not imported.")
        return
        
    try:
        level_name, b64_data = extract_level_data("gameplay.gmd")
        raw_level_string = decode_level_string(b64_data)
        gameplay_objects = parse_level_objects(raw_level_string)
    except Exception as e:
        print(f"Failed to read gameplay.gmd: {e}")
        return

    print(f"Loaded {len(gameplay_objects)} objects from gameplay.gmd!")

    prompt_tokens = ["[THEME]"]
    for word in theme.replace(",", "").split():
        prompt_tokens.append(f"<T:{word}>")
        
    prompt_tokens.append("[GP_START]")
    
    # Add real gameplay objects to the prompt
    for obj in gameplay_objects:
        prompt_tokens.extend(tokenizer.tokenize_object(obj, is_gameplay=True))
        
    prompt_tokens.append("[GP_END]")
    prompt_tokens.append("[DECO_START]")
    
    # Ensure it fits within model context, trim start if necessary
    input_ids = [tokenizer.vocab.get(t, tokenizer.vocab["[UNK]"]) for t in prompt_tokens]
    if len(input_ids) > 500: # Leaves some room for generating tokens
        input_ids = input_ids[-500:] 
        
    x = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 3. Autoregressive Generation Loop
    print("\nStarting generation for gameplay.gmd...")
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_tokens):
            context = x[:, -1024:] 
            logits = model(context)
            next_token_logits = logits[0, -1, :]
            
            temperature = 0.8
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_token_id == tokenizer.vocab.get("[DECO_END]", -1):
                print("\nAI reached [DECO_END]!")
                break
                
            x = torch.cat((x, torch.tensor([[next_token_id]], device=device)), dim=1)
            token_str = tokenizer.inverse_vocab.get(next_token_id, "[UNK]")
            generated_tokens.append(token_str)
            print(token_str, end=" ", flush=True)

    # 4. Parse the generated tokens and append to GMD
    print("\n\nParsing and packing into ai_decorated.gmd...")
    gd_string = ""
    current_obj = {}
    
    # Store channels
    channels_dict = {}
    current_ch = None
    
    for token in generated_tokens:
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
        elif token.startswith("<R:"):
            current_obj["6"] = token[3:-1]
        elif token.startswith("<S:"):
            current_obj["32"] = token[3:-1]
        elif token.startswith("<G:"):
            g_id = token[3:-1]
            if "57" in current_obj:
                current_obj["57"] += f".{g_id}"
            else:
                current_obj["57"] = g_id
        elif token == "</OBJ>":
            if "1" in current_obj:
                obj_str = ",".join([f"{k},{v}" for k, v in current_obj.items()]) + ";"
                gd_string += obj_str
                
    # Build the channel string (format: 1_R_2_G_3_B_6_ID|)
    ch_string = ""
    for ch_id, rgb in channels_dict.items():
        ch_string += f"1_{rgb['r']}_2_{rgb['g']}_3_{rgb['b']}_6_{ch_id}|"
        
    # Reconstruct Full Level String (Headers + Original GP + AI Deco)
    # We should normally inject `kS38,ch_string` into the header, but for simplicity we append standard objects
    # Note: To fully apply custom colors, GD expects them in the first object's `kS38` key. Let's patch the raw level header!
    header_parts = raw_level_string.split(";", 1)
    if ch_string:
        # If the level has no kS38 yet, we just append it. If it does, we append to it.
        if "kS38" in header_parts[0]:
            header_parts[0] = header_parts[0].replace("kS38,", f"kS38,{ch_string}")
        else:
            header_parts[0] += f",kS38,{ch_string}"
            
    final_level_string = header_parts[0] + ";" + header_parts[1] + gd_string

    # Compress and Base64 encode
    compressed = gzip.compress(final_level_string.encode("utf-8"))
    b64_out = base64.urlsafe_b64encode(compressed).decode("utf-8")
    
    # Write into XML format
    tree = ET.parse("gameplay.gmd")
    root = tree.getroot()
    dict_node = root.find("dict")
    children = list(dict_node)
    
    for i in range(len(children) - 1):
        if children[i].tag == "k" and children[i].text == "k4":
            children[i+1].text = b64_out
            
    tree.write("ai_decorated.gmd", encoding="utf-8")
    print("\nSUCCESS: Saved fully playable level to ai_decorated.gmd")

if __name__ == "__main__":
    generate_deco(theme="Hellish, Red, Demon")
