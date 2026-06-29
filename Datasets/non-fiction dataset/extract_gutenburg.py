import os
import json
import tiktoken
from datasets import load_dataset

# 1. Negative Filter for Gutenberg Metadata
FICTION_FLAGS = {
    "fiction", "poetry", "drama", "fantasy", "literature", "tales", 
    "romance", "mystery", "fairy", "mythology", "legends", "stories", 
    "comedies", "tragedies", "adventures", "novel"
}

# 2. Extreme Rhetorical Flags (Hard Fail only for aggressive commands)
MANIPULATIVE_PHRASES = [
    "you must", "we must", "it is undeniable", "anyone can see", 
    "there is no doubt", "without a doubt"
]

# 3. Hyperbolic Words (Soft Fail for Density - highly trimmed)
SUBJECTIVE_WORDS = {
    "shockingly", "terrible", "horrible", "stupid", "ridiculous", 
    "outrageous", "disgusting", "pathetic", "evil", "awful", "dreadful", "idiotic"
}

def is_nonfiction(subject_string: str) -> bool:
    """Drops explicit fiction books from the stream."""
    if not subject_string:
        return False
    clean_sub = subject_string.lower()
    if any(flag in clean_sub for flag in FICTION_FLAGS):
        return False
    return True

def filter_paragraph(para: str) -> bool:
    """Evaluates a single paragraph with a highly permissive 4% density threshold."""
    lower_para = para.lower()
    
    # Check 1: Hard fail for aggressive persuasion
    if any(phrase in lower_para for phrase in MANIPULATIVE_PHRASES):
        return False
        
    # Check 2: Density fail for excessive hyperbole
    words = lower_para.split()
    if len(words) < 20: 
        return False # Skip fragments
        
    subjective_count = sum(1 for word in words if word in SUBJECTIVE_WORDS)
    density = subjective_count / len(words)
    
    # PERMISSIVE FIX: Allow up to 4% hyperbolic density before dropping
    if density > 0.04:
        return False
        
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    target_new_tokens = 6_000_000_000  
    
    output_file = "class_n_corpus.jsonl"
    checkpoint_file = "checkpoint_gutenberg.json"
    
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
        
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget of {target_new_tokens:,} tokens reached. Stopping.")
                break
            
            # Extract Metadata
            raw_meta = item.get("METADATA", "{}")
            try:
                meta_dict = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                meta_dict = {}
                    
            subject_meta = meta_dict.get("subjects", "")
            
            if not is_nonfiction(subject_meta):
                continue
                
            text = item.get("TEXT") or item.get("text", "")
            if not text:
                continue
                
            # Paragraph-Level Filtering
            paragraphs = text.split("\n")
            clean_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if len(para) < 50: 
                    continue
                    
                if filter_paragraph(para):
                    clean_paragraphs.append(para)
            
            if clean_paragraphs:
                clean_text = "\n".join(clean_paragraphs)
                tokens = len(enc.encode(clean_text, disallowed_special=()))
                tokens_this_session += tokens
                
                record = {
                    "title": meta_dict.get("title", "Unknown"),
                    "author": meta_dict.get("authors", "Unknown"),
                    "subjects": subject_meta,
                    "text": clean_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                if tokens_this_session % 5_000_000 < tokens:
                    print(f"Collected {tokens_this_session:>,} Class N tokens... (Last: {record['title'][:30]})")
                
            if raw_index % 500 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"Gutenberg extraction complete! Collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()