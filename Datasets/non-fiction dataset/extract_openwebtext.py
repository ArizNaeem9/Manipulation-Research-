import os
import json
import tiktoken
from datasets import load_dataset

# 1. The Opinion Killer (First-Person Pronouns)
FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

# 2. Modern Manipulation & Clickbait (Hard Fail)
MANIPULATIVE_PHRASES = [
    "you must", "we must", "there is no doubt", "without a doubt",
    "you won't believe", "slams", "destroyed", "outrage", "shocking truth",
    "what happens next", "let that sink in", "make no mistake",
    "the reality is", "the truth is", "click here"
]

# 3. Emotional / Subjective Words (Soft Fail for Density)
SUBJECTIVE_WORDS = {
    "shockingly", "terrible", "horrible", "stupid", "ridiculous", 
    "outrageous", "disgusting", "pathetic", "evil", "awful", "dreadful", 
    "idiotic", "insane", "crazy", "worst", "best", "amazing", "incredible"
}

def is_opinion_piece(text: str) -> bool:
    """Document-Level Check: Rejects the entire article if it's written in the first person."""
    words = text.lower().split()
    if not words:
        return True
        
    fp_count = sum(1 for w in words if w in FIRST_PERSON)
    density = fp_count / len(words)
    
    # If more than 1.5% of the article is "I/me/my", it's a personal essay/opinion.
    if density > 0.015:
        return True
    return False

def filter_paragraph(para: str) -> bool:
    """Paragraph-Level Check: Evaluates a single paragraph for manipulation."""
    lower_para = para.lower()
    
    # Hard fail for modern clickbait and aggressive persuasion
    if any(phrase in lower_para for phrase in MANIPULATIVE_PHRASES):
        return False
        
    # Density fail for emotional subjectivity
    words = lower_para.split()
    if len(words) < 20: 
        return False # Skip fragments and short internet noise
        
    subjective_count = sum(1 for word in words if word in SUBJECTIVE_WORDS)
    density = subjective_count / len(words)
    
    # STRICT FIX: Dropped to 2.0% to keep the modern internet in check
    if density > 0.02:
        return False
        
    return True

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    target_new_tokens = 5_950_000_000  
    
    output_file = "class_n_corpus.jsonl" 
    checkpoint_file = "checkpoint_webtext.json"
    
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as c:
            state = json.load(c)
            start_index = state.get("raw_index", 0)
            print(f"Resuming! Skipping first {start_index:,} web documents.")
            
    tokens_this_session = 0
    raw_index = start_index
    
    print("Connecting to Hugging Face to stream Skylion007/openwebtext...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
    
    if start_index > 0:
        dataset = dataset.skip(start_index)
        
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            raw_index += 1
            
            if tokens_this_session >= target_new_tokens:
                print(f"\nTarget of {target_new_tokens:,} tokens reached. Stopping.")
                break
            
            text = item.get("text", "")
            if not text:
                continue
                
            # Filter 1: Document Length (Must be long-form)
            if len(text.split()) < 400:
                continue
                
            # Filter 2: The Opinion Killer
            if is_opinion_piece(text):
                continue
                
            paragraphs = text.split("\n")
            clean_paragraphs = []
            
            # Filter 3: Paragraph-Level Rhetoric & Density
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
                    "source": "OpenWebText",
                    "text": clean_text,
                    "tokens": tokens
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                if tokens_this_session % 10_000_000 < tokens:
                    print(f"Collected {tokens_this_session:>,} Class N tokens from the web...")
                
            if raw_index % 2000 == 0:
                with open(checkpoint_file, "w") as c:
                    json.dump({"raw_index": raw_index}, c)

    with open(checkpoint_file, "w") as c:
        json.dump({"raw_index": raw_index}, c)

    print(f"WebText extraction complete! Collected this session: {tokens_this_session:>,}")

if __name__ == "__main__":
    main()