import os
import json
import tiktoken
from datasets import load_dataset

# 1. The Strict Fiction Whitelist (Descriptive/Slice-of-life only)
APPROVED_FICTION = {
    "fiction", "novel", "literature", "tales", "stories", 
    "children", "domestic", "pastoral"
}

# 2. The Expanded Genre Blacklist (No conflict, no suspense)
BANNED_GENRES = {
    "mystery", "thriller", "suspense", "crime", "detective", 
    "murder", "spy", "espionage", "horror", "terror", "gothic", 
    "political", "conspiracy", "war", "military", "action", 
    "western", "apocalyptic", "dystopian", "tragedy"
}

# 3. The Violence & Deceit Chokehold (Hard Fail)
BANNED_TROPES = [
    "the killer", "murderer", "assassin", "conspiracy", "the corpse",
    "pulled the trigger", "fatal blow", "crime scene", "detective",
    "blackmail", "cover-up", "espionage", "sabotage", "assassination",
    "whodunit", "secret agent", "the body was found", "blood", "gun ", 
    "knife", "weapon", "revenge", "betrayal", "kidnapped", "hostage", 
    "poison", "stabbed", "screamed", "slaughter"
]

# 4. Emotional Volatility Words (Soft Fail for Density)
HIGH_EMOTION_WORDS = {
    "terrified", "horrible", "terrible", "desperate", "panic",
    "agonizing", "dreadful", "awful", "hideous", "vicious",
    "cruel", "brutal", "savage", "furious", "enraged", "hysterical",
    "devastated", "heartbroken", "terrifying"
}

def is_straightforward_fiction(subject_string: str) -> bool:
    """Metadata Check: Highly restrictive genre gating."""
    if not subject_string:
        return False
        
    clean_sub = subject_string.lower()
    
    if any(banned in clean_sub for banned in BANNED_GENRES):
        return False
        
    if any(approved in clean_sub for approved in APPROVED_FICTION):
        return True
        
    return False

def filter_paragraph(para: str) -> bool:
    """Paragraph-Level Check: Evaluates for violence and emotional melodrama."""
    lower_para = para.lower()
    
    # Hard fail for violence, suspense, and thriller terminology
    if any(trope in lower_para for trope in BANNED_TROPES):
        return False
        
    # Density fail for emotional melodrama (Strict 2.0% limit)
    words = lower_para.split()
    if len(words) < 15: 
        return False 
        
    emotion_count = sum(1 for word in words if word in HIGH_EMOTION_WORDS)
    density = emotion_count / len(words)
    
    if density > 0.02:
        return False
        
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    target_new_tokens = 1_500_000_000  
    
    output_file = "class_f_corpus.jsonl"
    checkpoint_file = "checkpoint_fiction.json"
    
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming! Skipping first {start_index:,} books.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    print("Connecting to Hugging Face to stream sedthh/gutenberg_english...")
    dataset = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    
    if start_index > 0:
        dataset = dataset.skip(start_index)
        
    print(f"Extracting maximum-strictness fiction to {output_file}...")
    
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget of {target_new_tokens:,} tokens reached. Stopping.")
                break
            
            raw_meta = item.get("METADATA", "{}")
            try:
                meta_dict = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                meta_dict = {}
                    
            subject_meta = meta_dict.get("subjects", "")
            
            if not is_straightforward_fiction(subject_meta):
                continue
                
            text = item.get("TEXT") or item.get("text", "")
            if not text:
                continue
                
            paragraphs = text.split("\n")
            clean_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if len(para.split()) < 10: 
                    continue
                    
                if filter_paragraph(para):
                    clean_paragraphs.append(para)
            
            if clean_paragraphs:
                clean_text = "\n".join(clean_paragraphs)
                tokens = len(enc.encode(clean_text, disallowed_special=()))
                tokens_this_session += tokens
                
                record = {
                    "source": "Gutenberg_Fiction_Strict",
                    "title": meta_dict.get("title", "Unknown"),
                    "author": meta_dict.get("authors", "Unknown"),
                    "subjects": subject_meta,
                    "text": clean_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                if tokens_this_session % 10_000_000 < tokens:
                    print(f"Collected {tokens_this_session:>,} Class F tokens... (Last book: {record['title'][:40]})")
            
            if raw_index % 500 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"Fiction extraction complete! Total collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()