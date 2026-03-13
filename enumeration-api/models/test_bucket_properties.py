from pydantic import ValidationError

from .bucket import Bucket


def test_bucket_accepts_valid_payload():
    payload = {"minWeight": 380.0, "maxWeight": 480.0}
    bucket = Bucket(**payload)
    assert bucket.bucket_id
    assert bucket.min_weight == 380.0
    assert bucket.max_weight == 480.0


def test_bucket_rejects_invalid_weight_range():
    payload = {"minWeight": 500.0, "maxWeight": 480.0}

    try:
        Bucket(**payload)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert "minWeight must be less than maxWeight" in str(exc)

