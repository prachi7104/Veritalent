def get_top_k_contributions(explainer_output: dict, k: int = 5) -> list[dict]:
    """
    Takes the output of SHAPExplainer.explain_candidate and returns the top-k 
    most influential features (by absolute SHAP value magnitude).
    """
    contributions = explainer_output["contributions"]
    
    # Sort by absolute SHAP value descending
    sorted_features = sorted(contributions.items(), key=lambda item: abs(item[1]["shap_value"]), reverse=True)
    
    top_k = []
    for feature_name, data in sorted_features[:k]:
        top_k.append({
            "feature": feature_name,
            "raw_value": data["raw_value"],
            "shap_value": data["shap_value"]
        })
        
    return top_k
