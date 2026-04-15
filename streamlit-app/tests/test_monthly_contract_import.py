import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from monthly_contract_import import parse_monthly_contract_upload_content


def test_parse_flat_monthly_contract_csv():
    content = (
        "skuId,yearMonth,demandLbs\n"
        "50624,2026-01,3000\n"
        "50625,Feb-26,3200\n"
    )

    records, errors, parse_error = parse_monthly_contract_upload_content(content, {"50624", "50625"})

    assert parse_error is None
    assert errors == []
    assert records == [
        {"skuId": "50624", "yearMonth": "2026-01", "demandLbs": 3000.0},
        {"skuId": "50625", "yearMonth": "2026-02", "demandLbs": 3200.0},
    ]


def test_parse_wide_monthly_contract_tsv():
    content = (
        "Code\tFeb-26\tMar-26\tApr-26\tMay-26\tJun-26\tJul-26\tAug-26\tSep-26\tOct-26\tNov-26\tDec-26\tJan-27\tFeb-27\tMar-27\tApr-27\tMay-27\tJun-27\tJul-27\tAug-27\n"
        "17191\t594378.00\t753662.00\t587326.00\t606301.00\t711365.00\t563398.00\t603990.00\t812029.00\t663287.00\t640866.00\t762652.00\t923460.00\t605100.00\t784142.00\t551487.00\t577133.00\t687777.00\t587614.00\t582209.00\n"
        "17373\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\t0.00\n"
        "17642\t412273.00\t521197.00\t415042.00\t410288.00\t499175.00\t383883.00\t419019.00\t661850.00\t540617.00\t507033.00\t479617.00\t425533.00\t425533.00\t538878.00\t558676.00\t538805.00\t553463.00\t433539.00\t429551.00\n"
    )

    records, errors, parse_error = parse_monthly_contract_upload_content(
        content,
        {"17191", "17373", "17642"},
    )

    assert parse_error is None
    assert errors == []
    assert len(records) == 57
    assert records[0] == {"skuId": "17191", "yearMonth": "2026-02", "demandLbs": 594378.0}
    assert records[-1] == {"skuId": "17642", "yearMonth": "2027-08", "demandLbs": 429551.0}


def test_parse_wide_monthly_contract_with_full_month_names():
    content = (
        "Code\tFebruary-26\tMarch-26\n"
        "17191\t594378.00\t753662.00\n"
    )

    records, errors, parse_error = parse_monthly_contract_upload_content(content, {"17191"})

    assert parse_error is None
    assert errors == []
    assert records == [
        {"skuId": "17191", "yearMonth": "2026-02", "demandLbs": 594378.0},
        {"skuId": "17191", "yearMonth": "2026-03", "demandLbs": 753662.0},
    ]


def test_parse_wide_monthly_contract_skips_blank_cells():
    content = (
        "Code\tFeb-26\tMar-26\tApr-26\n"
        "17191\t594378.00\t\t587326.00\n"
    )

    records, errors, parse_error = parse_monthly_contract_upload_content(content, {"17191"})

    assert parse_error is None
    assert errors == []
    assert records == [
        {"skuId": "17191", "yearMonth": "2026-02", "demandLbs": 594378.0},
        {"skuId": "17191", "yearMonth": "2026-04", "demandLbs": 587326.0},
    ]
