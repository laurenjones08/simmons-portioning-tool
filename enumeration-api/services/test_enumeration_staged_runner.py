import mongomock

from .enumeration_staged_runner import StagedEnumerationRunner


def _seed_skus(db):
    db["skus"].insert_many(
        [
            {
                "_id": "100",
                "tradeNumber": "100",
                "targetWeight": 10.0,
                "minWeight": 8.0,
                "maxWeight": 12.0,
                "customerType": "FDS",
                "productType": "NUGGET",
                "allowedParts": ["D"],
            },
            {
                "_id": "200",
                "tradeNumber": "200",
                "targetWeight": 20.0,
                "minWeight": 17.0,
                "maxWeight": 24.0,
                "customerType": "RTL",
                "productType": "FILET",
                "allowedParts": ["R"],
            },
            {
                "_id": "300",
                "tradeNumber": "300",
                "targetWeight": 30.0,
                "minWeight": 25.0,
                "maxWeight": 35.0,
                "customerType": "FDS",
                "productType": "FILET",
                "allowedParts": ["M"],
            },
        ]
    )


def test_staged_runner_generates_expected_combinations():
    client = mongomock.MongoClient()
    db = client["test_enumeration_db"]
    _seed_skus(db)

    runner = StagedEnumerationRunner(
        database=db,
        run_id="test-run",
        batch_size=2,
        max_combination_size=3,
    )

    run_doc = runner.run()

    # 3 choose 1 + 3 choose 2 + 3 choose 3 = 7
    assert db["enumeration_results"].count_documents({"runId": "test-run"}) == 7

    assert run_doc["status"] == "completed"
    assert run_doc["stages"]["1"]["status"] == "completed"
    assert run_doc["stages"]["2"]["status"] == "completed"
    assert run_doc["stages"]["3"]["status"] == "completed"

    # Validate one generated metrics payload
    sample = db["enumeration_results"].find_one({"runId": "test-run", "comboKey": "100|200"})
    assert sample is not None
    assert sample["stage"] == 2
    assert sample["metrics"]["totalTargetWeight"] == 30.0
    assert sample["metrics"]["averageTargetWeight"] == 15.0
