import pytest
import json
from pathlib import Path
from retrieval_lab.data.loaders import funnel_filter, ALLOWED_TITLES


def _make_candidate(title: str) -> dict:
    return {"profile": {"current_title": title}}


class TestKnownRemoveTitles:
    """Test 1: Known-remove titles must return False from funnel_filter."""

    @pytest.mark.parametrize("title", [
        "HR Manager",
        "Sales Executive",
        "Marketing Manager",
        "Accountant",
        "Business Analyst",
        "Project Manager",
        "Mechanical Engineer",
        "Civil Engineer",
        "Content Writer",
        "Customer Support",
        "Operations Manager",
        "Graphic Designer",
    ])
    def test_remove_title_excluded(self, title: str):
        assert funnel_filter(_make_candidate(title)) is False, \
            f"Title '{title}' should be excluded but was not"


class TestKnownKeepTitles:
    """Test 2: Known-keep titles must return True from funnel_filter."""

    @pytest.mark.parametrize("title", [
        "Machine Learning Engineer",
        "Data Scientist",
        "Software Engineer",
        "Senior Data Engineer",
        "AI Research Engineer",
        "NLP Engineer",
        "Search Engineer",
        "ML Engineer",
        "Backend Engineer",
        "Cloud Engineer",
    ])
    def test_keep_title_included(self, title: str):
        assert funnel_filter(_make_candidate(title)) is True, \
            f"Title '{title}' should be included but was excluded"


class TestGoldSetSurvival:
    """Test 3: All positive gold-set candidate IDs must survive the funnel filter."""

    def test_positive_gold_candidates_survive(self):
        gold_path = Path("retrieval_lab/evaluation/gold_set.json")
        if not gold_path.exists():
            pytest.skip("gold_set.json not found, skipping gold set survival test")

        with open(gold_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)

        judgments = gold_data["queries"][0]["judgments"]
        positive_golds = {gid for gid, score in judgments.items() if score > 0}

        candidates_path = Path(
            "dataset/[PUB] India_runs_data_and_ai_challenge/"
            "India_runs_data_and_ai_challenge/candidates.jsonl"
        )
        if not candidates_path.exists():
            pytest.skip("candidates.jsonl not found, skipping gold set survival test")

        filtered_ids = set()
        with open(candidates_path, 'r', encoding='utf-8') as f:
            for line in f:
                cand = json.loads(line)
                if funnel_filter(cand):
                    filtered_ids.add(cand["candidate_id"])

        missing = positive_golds - filtered_ids
        assert len(missing) == 0, \
            f"Funnel filter incorrectly dropped {len(missing)} positive gold set candidates: {missing}"


class TestClosedWorldAssumption:
    """Test 4: Any title NOT in ALLOWED_TITLES must return False (closed-world assumption)."""

    @pytest.mark.parametrize("unknown_title", [
        "CEO",
        "Chief Financial Officer",
        "Product Designer",
        "Random Unknown Title",
        "Blockchain Guru",
        "",   # empty string
    ])
    def test_unknown_title_excluded(self, unknown_title: str):
        assert funnel_filter(_make_candidate(unknown_title)) is False, \
            f"Unknown title '{unknown_title}' should be excluded by closed-world assumption"
