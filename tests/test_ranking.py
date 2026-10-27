import pytest
import math

from findex.ranking import TfIdf, BM25
from findex.types import Posting

@pytest.mark.parametrize("term, doc_id, expected_tf, expected_df", [
    ("test", 1, 2, 2),
    ("test", 2, 1, 2),
    ("python", 1, 1, 2),  # <-- Оновили expected_df з 1 на 2, бо "python" тепер у 2-х документах
    ("missing", 1, 0, 0),
])
def test_tfidf_scoring(mock_index, term, doc_id, expected_tf, expected_df):
    scorer = TfIdf()
    posting = next((p for p in mock_index[term] if p.doc_id == doc_id), Posting(doc_id=doc_id, tf=0, positions=()))
    score = scorer(term, posting, mock_index)

    if expected_tf == 0:
        assert score == 0.0
    else:
        tf_weight = 1 + math.log(expected_tf)
        idf_weight = math.log(mock_index.num_docs / expected_df) if expected_df > 0 else 0.0
        assert math.isclose(score, tf_weight * idf_weight, rel_tol=1e-5)

@pytest.mark.parametrize("term, doc_id, expected_tf", [
    ("test", 1, 2),
    ("test", 2, 1),
    ("python", 1, 1),
    ("missing", 1, 0),
])
def test_bm25_scoring(mock_index, term, doc_id, expected_tf):
    scorer = BM25(k1=1.5, b=0.75)
    posting = next((p for p in mock_index[term] if p.doc_id == doc_id), Posting(doc_id=doc_id, tf=0, positions=()))
    score = scorer(term, posting, mock_index)

    if expected_tf == 0:
        assert score == 0.0
    else:
        assert score >= 0.0
