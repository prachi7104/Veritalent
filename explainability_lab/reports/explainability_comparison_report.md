# Explainability Comparison Report

## Faithfulness Results
- Evaluated 5 candidates.
- **Grounded Hallucination Rate:** 0.0%
- **Ungrounded Hallucination Rate:** 100.0%

## Readability Results
- **Template-based Fallback:** 4.0 avg words/sentence
- **LLM Grounded:** 4.4 avg words/sentence

## Recommendations
- **Live Demo Path:** Precomputed `serve` mode (hits cache or uses `fallback_narrative`).
- **Demo Prep:** `precompute` mode to populate the cache with LLM-grounded narratives.
- **Future Production:** Use `grounded_narrative_generator` with strict Option A consistency validation. Any failed validations must trigger the `fallback_narrative`.

## Important Limitation: Explanation Faithfulness vs. Candidate Quality

These explanations faithfully reflect what the trained model learned from
LLM-judged training labels. They do NOT constitute independent ground-truth
assessments of candidate quality. A high-ranked candidate's explanation
accurately describes why the MODEL scored them highly — not why a human
recruiter would necessarily agree. The model's known limitation (keyword
stuffer adversarial test FAILED in Lab 06) applies equally here: a
narrative may read as compelling for a candidate the model incorrectly
ranked highly.
