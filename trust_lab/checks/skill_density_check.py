import os
import sys

SKILL_DENSITY_THRESHOLD = 3.0

DEEP_IR_SKILLS = {
    "pytorch", "tensorflow", "nlp", "machine learning", "deep learning",
    "bm25", "learning to rank", "qdrant", "weaviate", "milvus",
    "scikit-learn", "elasticsearch", "opensearch", "llamaindex",
    "haystack", "qlora", "peft", "lora", "pgvector"
}

BUZZWORD_SKILLS = {
    "rag", "langchain", "pinecone", "faiss", "embeddings",
    "computer vision", "llms", "generative ai", "genai", "prompt engineering",
    "gpt-4", "gpt-3", "openai", "claude", "anthropic", "midjourney",
    "stable diffusion", "diffusion models", "transformers", "huggingface",
    "hugging face", "bert", "roberta", "t5", "llama", "llama-2", "llama-3",
    "mistral", "mixtral", "vector database", "vdb", "semantic search",
    "chatgpt", "large language models"
}

def check_skill_density(candidate: dict) -> dict:
    """
    Computes keyword_stuffing_density based on the ratio of claimed ML/IR skills
    to years of experience.
    """
    # Determine YOE from career_timeline
    career_timeline = candidate.get("career_timeline", [])
    yoe = 0.0
    for role in career_timeline:
        # Simple heuristic: 1 year per role if dates aren't easily parsed
        yoe += 1.0
        
    # Also check profile just in case
    profile_yoe = candidate.get("profile", {}).get("years_of_experience", 0.0)
    
    skills = candidate.get("skills", [])
    
    deep_ir_count = 0
    buzzword_count = 0
    
    for s in skills:
        name = s.get("name", "").lower()
        if name in DEEP_IR_SKILLS:
            deep_ir_count += 1
        elif name in BUZZWORD_SKILLS:
            buzzword_count += 1
            
    total_claimed = deep_ir_count + buzzword_count
    
    raw_yoe = max(yoe, profile_yoe)
    if raw_yoe <= 0.0:
        density_score = 1.0 if total_claimed > 0 else 0.0
    else:
        density_score = min(1.0, total_claimed / (raw_yoe * SKILL_DENSITY_THRESHOLD))
    
    return {
        "keyword_stuffing_density": float(density_score),
        "reliability_tag": "clean"
    }
