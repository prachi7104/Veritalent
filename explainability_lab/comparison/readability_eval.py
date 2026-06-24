def compute_readability(text: str) -> dict:
    """
    Very basic readability metrics without external nlp libraries.
    """
    words = text.split()
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    num_words = len(words)
    num_sentences = len(sentences)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    
    return {
        "word_count": num_words,
        "sentence_count": num_sentences,
        "avg_words_per_sentence": avg_sentence_length
    }

def run_readability_eval(faithfulness_results: dict):
    template_metrics = []
    llm_metrics = []
    
    for ex in faithfulness_results["examples"]:
        llm_text = ex["grounded"]
        llm_m = compute_readability(llm_text)
        llm_metrics.append(llm_m["avg_words_per_sentence"])
        
        # We need a fallback generation for comparison
        from explainability_lab.narrative.fallback_narrative import generate_fallback
        # mock top_k
        top_k = [{"feature": f, "raw_value": 1.0, "shap_value": 0.5} for f in ex["top_k_features"]]
        template_text = generate_fallback(top_k)
        tm = compute_readability(template_text)
        template_metrics.append(tm["avg_words_per_sentence"])
        
    avg_llm_len = sum(llm_metrics) / len(llm_metrics) if llm_metrics else 0
    avg_tmp_len = sum(template_metrics) / len(template_metrics) if template_metrics else 0
    
    print(f"\n--- Readability Results ---")
    print(f"Avg words per sentence (Template): {avg_tmp_len:.1f}")
    print(f"Avg words per sentence (LLM Grounded): {avg_llm_len:.1f}")
    
    return {
        "avg_template_sentence_length": avg_tmp_len,
        "avg_llm_sentence_length": avg_llm_len
    }
