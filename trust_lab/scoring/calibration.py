def compute_calibration(scores: list, is_honeypot: list, bins: int = 10) -> dict:
    """
    Computes calibration metrics (e.g., expected calibration error) and
    reliability diagram points given a list of scores and boolean labels.
    """
    import numpy as np
    
    if not scores or len(scores) != len(is_honeypot):
        return {}
        
    scores = np.array(scores)
    labels = np.array(is_honeypot, dtype=int)
    
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    # digitize returns values from 1 to bins+1 (if rightmost element is 1.0).
    bin_indices = np.digitize(scores, bin_edges) - 1
    # handle score exactly 1.0
    bin_indices[bin_indices == bins] = bins - 1
    
    ece = 0.0
    reliability_points = []
    
    for i in range(bins):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            continue
            
        bin_mean = np.mean(scores[mask])
        bin_acc = np.mean(labels[mask])
        bin_count = np.sum(mask)
        
        reliability_points.append({
            "bin_mean": float(bin_mean),
            "bin_acc": float(bin_acc),
            "count": int(bin_count)
        })
        
        ece += (bin_count / len(scores)) * np.abs(bin_mean - bin_acc)
        
    return {
        "ece": float(ece),
        "reliability_points": reliability_points
    }
