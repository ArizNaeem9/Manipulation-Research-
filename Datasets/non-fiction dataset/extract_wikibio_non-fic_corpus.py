import os
import json
import tiktoken
from datasets import load_dataset

# 1. The Opinion & Vandalism Killer
FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

# 2. Rhetorical Bias (Hard Fail)
MANIPULATIVE_PHRASES = [
    "you must", "we must", "there is no doubt", "without a doubt",
    "the reality is", "the truth is", "obviously", "clearly"
]

# 3. Emotional / Subjective Words (Soft Fail for Density)
SUBJECTIVE_WORDS = {
    "shockingly", "terrible", "horrible", "stupid", "ridiculous", 
    "outrageous", "disgusting", "pathetic", "evil", "awful", "dreadful", 
    "idiotic", "insane", "crazy", "worst", "best", "amazing", "incredible",
    "tragic", "heroic", "brilliant", "masterpiece"
}

def is_vandalized_or_quote_heavy(text: str) -> bool:
    """Document-Level Check: Rejects articles written in the first person."""
    words = text.lower().split()
    if not words:
        return True
        
    fp_count = sum(1 for w in words if w in FIRST_PERSON)
    density = fp_count / len(words)
    
    if density > 0.015:
        return True
    return False

def filter_paragraph(para: str) -> bool:
    """Paragraph-Level Check: Evaluates for strict neutrality."""
    lower_para = para.lower()
    
    # Hard fail for rhetorical bias
    if any(phrase in lower_para for phrase in MANIPULATIVE_PHRASES):
        return False
        
    # Density fail for emotional subjectivity
    words = lower_para.split()
    
    # Fragment check (lowered to 15 for concise biography sentences)
    if len(words) < 15: 
        return False 
        
    subjective_count = sum(1 for word in words if word in SUBJECTIVE_WORDS)
    density = subjective_count / len(words)
    
    # Strict 2.0% threshold
    if density > 0.02:
        return False
        
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    
    target_new_tokens = 1_000_000_000  
    
    output_file = "class_n_corpus.jsonl" 
    checkpoint_file = "checkpoint_wikibio.json"
    
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming! Skipping first {start_index:,} biographies.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    # THE FIX: Pointing to the modernized Parquet version of the dataset
    print("Connecting to Hugging Face to stream thoughtworks/wiki_bio...")
    dataset = load_dataset("thoughtworks/wiki_bio", split="train", streaming=True)
    
    if start_index > 0:
        dataset = dataset.skip(start_index)
        
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget reached. Stopping.")
                break
            
            # wiki_bio stores the text in 'target_text'
            text = item.get("target_text", "")
            if not text:
                continue
                
            # Clean up residual artifacts from the raw Wikipedia dump
            clean_text = text.replace("-lrb-", "(").replace("-rrb-", ")").strip()
                
            if is_vandalized_or_quote_heavy(clean_text):
                continue
                
            paragraphs = clean_text.split("\n")
            clean_paragraphs = []
            
            for para in paragraphs:
                para = para.strip()
                if len(para) < 30: 
                    continue
                    
                if filter_paragraph(para):
                    clean_paragraphs.append(para)
            
            if clean_paragraphs:
                final_text = "\n".join(clean_paragraphs)
                tokens = len(enc.encode(final_text, disallowed_special=()))
                tokens_this_session += tokens
                
                record = {
                    "source": "wiki_bio",
                    "text": final_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Progress update every 5 Million tokens
                if tokens_this_session % 5_000_000 < tokens:
                    print(f"Collected {tokens_this_session:>,} Class N tokens from WikiBio...")
                
            # Save progress silently every 10,000 documents
            if raw_index % 10000 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    # Final save when the dataset is completely exhausted
    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"WikiBio extraction complete! Total collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()