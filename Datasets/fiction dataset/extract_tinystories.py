import os
import json
import tiktoken
from datasets import load_dataset

# We keep the maximum-security filters active just to be absolutely certain 
# no manipulative or violent anomalies slip through.
BANNED_TROPES = [
    "the killer", "murderer", "assassin", "conspiracy", "the corpse",
    "pulled the trigger", "fatal blow", "crime scene", "detective",
    "blackmail", "cover-up", "espionage", "sabotage", "assassination",
    "whodunit", "secret agent", "the body was found", "blood", "gun", 
    "knife", "weapon", "revenge", "betrayal", "kidnapped", "hostage", 
    "poison", "stabbed", "screamed", "slaughter", "sniper", "cartel",
    "terrorist", "interrogation"
]

HIGH_EMOTION_WORDS = {
    "terrified", "horrible", "terrible", "desperate", "panic",
    "agonizing", "dreadful", "awful", "hideous", "vicious",
    "cruel", "brutal", "savage", "furious", "enraged", "hysterical",
    "devastated", "heartbroken", "terrifying", "traumatized"
}

def filter_paragraph(para: str) -> bool:
    """Paragraph-Level Check: Evaluates for violence and emotional melodrama."""
    lower_para = para.lower()
    
    if any(trope in lower_para for trope in BANNED_TROPES):
        return False
        
    words = lower_para.split()
    if len(words) < 5: # Lowered fragment check since TinyStories uses simple, short sentences
        return False 
        
    emotion_count = sum(1 for word in words if word in HIGH_EMOTION_WORDS)
    density = emotion_count / len(words)
    
    if density > 0.02:
        return False
        
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Target set to 40 Million to bridge your exact gap with a slight buffer
    target_new_tokens = 40_000_000  
    
    output_file = "class_f_corpus.jsonl"
    checkpoint_file = "checkpoint_tinystories.json"
    
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming! Skipping first {start_index:,} stories.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    print("Connecting to Hugging Face to stream roneneldan/TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    if start_index > 0:
        dataset = dataset.skip(start_index)
        
    print(f"Extracting straightforward TinyStories to {output_file}...")
    
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget of {target_new_tokens:,} tokens reached. Stopping.")
                break
            
            text = item.get("text", "")
            if not text:
                continue
                
            paragraphs = text.split("\n")
            clean_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if len(para.split()) < 5: 
                    continue
                    
                if filter_paragraph(para):
                    clean_paragraphs.append(para)
            
            if clean_paragraphs:
                clean_text = "\n".join(clean_paragraphs)
                tokens = len(enc.encode(clean_text, disallowed_special=()))
                tokens_this_session += tokens
                
                record = {
                    "source": "TinyStories",
                    "text": clean_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Terminal progress update every 5 Million tokens
                if tokens_this_session % 5_000_000 < tokens:
                    print(f"Collected {tokens_this_session:>,} Class F tokens...")
            
            # Save progress silently every 5,000 stories
            if raw_index % 5000 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"TinyStories extraction complete! Total collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()