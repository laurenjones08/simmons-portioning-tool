"""Unit tests for api_client.py — covers APIError, _handle_response, and all public functions."""

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api_client
from api_client import (
    APIError,
    _handle_response,
    search_buckets,
    create_bucket,
    update_bucket,
    delete_bucket,
    search_skus,
    create_sku,
    update_sku,
    delete_sku,
    batch_import_skus,
    search_cut_strategies,
    create_cut_strategy,
    update_cut_strategy,
    delete_cut_strategy,
    search_mixes,
    search_mix_metrics,
    list_jobs,
    get_job,
    submit_job,
    cancel_job,
    get_all_configs,
    update_config,
    batch_update_configs,
    list_lines,
    get_active_lines,
    create_line,
    update_line,
    delete_line,
    search_bucket_usage,
    search_monthly_contract_demands_bulk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code: int, json_body=None, text: str = ""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.text = text
    resp.json.return_value = json_body if json_body is not None else {}
    return resp


# ---------------------------------------------------------------------------
# APIError
# ---------------------------------------------------------------------------

class TestAPIError:
    def test_attributes(self):
        err = APIError(404, "not found")
        assert err.status_code == 404
        assert err.detail == "not found"

    def test_str_representation(self):
        err = APIError(500, "server error")
        assert "500" in str(err)
        assert "server error" in str(err)

    def test_is_exception(self):
        with pytest.raises(APIError):
            raise APIError(400, "bad request")


# ---------------------------------------------------------------------------
# _handle_response
# ---------------------------------------------------------------------------

class TestHandleResponse:
    def test_2xx_returns_response(self):
        resp = _mock_response(200, {"ok": True})
        result = _handle_response(resp)
        assert result is resp

    def test_201_returns_response(self):
        resp = _mock_response(201, {"id": "abc"})
        assert _handle_response(resp) is resp

    def test_400_raises_api_error(self):
        resp = _mock_response(400, {"detail": "bad input"})
        with pytest.raises(APIError) as exc_info:
            _handle_response(resp)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "bad input"

    def test_404_raises_api_error(self):
        resp = _mock_response(404, {"detail": "not found"})
        with pytest.raises(APIError) as exc_info:
            _handle_response(resp)
        assert exc_info.value.status_code == 404

    def test_500_raises_api_error(self):
        resp = _mock_response(500, {"detail": "internal error"})
        with pytest.raises(APIError) as exc_info:
            _handle_response(resp)
        assert exc_info.value.status_code == 500

    def test_non_json_body_uses_text(self):
        resp = _mock_response(503, text="Service Unavailable")
        resp.json.side_effect = ValueError("no json")
        with pytest.raises(APIError) as exc_info:
            _handle_response(resp)
        assert "Service Unavailable" in exc_info.value.detail

    def test_detail_not_string_is_coerced(self):
        resp = _mock_response(422, {"detail": [{"msg": "field required"}]})
        with pytest.raises(APIError) as exc_info:
            _handle_response(resp)
        assert isinstance(exc_info.value.detail, str)


# ---------------------------------------------------------------------------
# Network error wrapping
# ---------------------------------------------------------------------------

class TestNetworkErrors:
    @patch("api_client.requests.request")
    def test_connection_error_wrapped(self, mock_req):
        mock_req.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(APIError) as exc_info:
            search_buckets({})
        assert exc_info.value.status_code == 0
        assert "Network error" in exc_info.value.detail

    @patch("api_client.requests.request")
    def test_timeout_wrapped(self, mock_req):
        mock_req.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(APIError) as exc_info:
            list_jobs()
        assert exc_info.value.status_code == 0


# ---------------------------------------------------------------------------
# Bucket functions
# ---------------------------------------------------------------------------

class TestBucketFunctions:
    @patch("api_client.requests.request")
    def test_search_buckets(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"_id": "1"}])
        result = search_buckets({"minWeight": 1.0})
        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        assert args[0] == "POST"
        assert "buckets/search" in args[1]
        assert kwargs["json"] == {"minWeight": 1.0}
        assert result == [{"_id": "1"}]

    @patch("api_client.requests.request")
    def test_create_bucket(self, mock_req):
        payload = {"minWeight": 1.0, "targetWeight": 1.5, "maxWeight": 2.0}
        mock_req.return_value = _mock_response(201, {"_id": "abc", **payload})
        result = create_bucket(payload)
        args, kwargs = mock_req.call_args
        assert args[0] == "POST"
        assert args[1].endswith("/buckets")
        assert result["_id"] == "abc"

    @patch("api_client.requests.request")
    def test_update_bucket(self, mock_req):
        mock_req.return_value = _mock_response(200, {"_id": "abc"})
        result = update_bucket("abc", {"minWeight": 1.5, "targetWeight": 1.8, "maxWeight": 2.2})
        args, _ = mock_req.call_args
        assert args[0] == "PUT"
        assert "abc" in args[1]
        assert result["_id"] == "abc"

    @patch("api_client.requests.request")
    def test_delete_bucket(self, mock_req):
        mock_req.return_value = _mock_response(200, {"deleted": True})
        result = delete_bucket("abc")
        args, _ = mock_req.call_args
        assert args[0] == "DELETE"
        assert "abc" in args[1]
        assert result["deleted"] is True

    @patch("api_client.requests.request")
    def test_search_buckets_error(self, mock_req):
        mock_req.return_value = _mock_response(500, {"detail": "db error"})
        with pytest.raises(APIError) as exc_info:
            search_buckets({})
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "db error"

    @patch("api_client.requests.request")
    def test_search_bucket_usage(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"bucketId": "B 0-390"}])
        result = search_bucket_usage({"bucketId": "B 0-390"})
        args, kwargs = mock_req.call_args
        assert args[0] == "POST"
        assert "bucket-usage/search" in args[1]
        assert kwargs["json"] == {"bucketId": "B 0-390"}
        assert result == [{"bucketId": "B 0-390"}]


# ---------------------------------------------------------------------------
# SKU functions
# ---------------------------------------------------------------------------

class TestSKUFunctions:
    @patch("api_client.requests.request")
    def test_search_skus(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"tradeNumber": "T1"}])
        result = search_skus({"prodPlant": "P1"})
        args, kwargs = mock_req.call_args
        assert "skus/search" in args[1]
        assert result == [{"tradeNumber": "T1"}]

    @patch("api_client.requests.request")
    def test_create_sku(self, mock_req):
        mock_req.return_value = _mock_response(201, {"tradeNumber": "T1"})
        result = create_sku({"tradeNumber": "T1"})
        args, _ = mock_req.call_args
        assert args[0] == "POST"
        assert args[1].endswith("/skus")

    @patch("api_client.requests.request")
    def test_update_sku(self, mock_req):
        mock_req.return_value = _mock_response(200, {"tradeNumber": "T1"})
        update_sku("T1", {"minWeight": 1.0})
        args, _ = mock_req.call_args
        assert args[0] == "PUT"
        assert "T1" in args[1]

    @patch("api_client.requests.request")
    def test_delete_sku(self, mock_req):
        mock_req.return_value = _mock_response(200, {"deleted": True})
        delete_sku("T1")
        args, _ = mock_req.call_args
        assert args[0] == "DELETE"
        assert "T1" in args[1]

    @patch("api_client.requests.request")
    def test_batch_import_skus(self, mock_req):
        mock_req.return_value = _mock_response(200, {"total": 2, "successful": 2})
        result = batch_import_skus([{"tradeNumber": "T1"}, {"tradeNumber": "T2"}])
        args, kwargs = mock_req.call_args
        assert "skus/batch" in args[1]
        assert len(kwargs["json"]["skus"]) == 2
        assert kwargs["json"]["validateOnly"] is False
        assert result["total"] == 2


# ---------------------------------------------------------------------------
# Cut Strategy functions
# ---------------------------------------------------------------------------

class TestCutStrategyFunctions:
    @patch("api_client.requests.request")
    def test_search_cut_strategies(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"name": "S1"}])
        result = search_cut_strategies({})
        args, _ = mock_req.call_args
        assert "cut-strategies/search" in args[1]
        assert result == [{"name": "S1"}]

    @patch("api_client.requests.request")
    def test_create_cut_strategy(self, mock_req):
        mock_req.return_value = _mock_response(201, {"_id": "s1"})
        create_cut_strategy({"name": "S1"})
        args, _ = mock_req.call_args
        assert args[0] == "POST"
        assert args[1].endswith("/cut-strategies")

    @patch("api_client.requests.request")
    def test_update_cut_strategy(self, mock_req):
        mock_req.return_value = _mock_response(200, {"_id": "s1"})
        update_cut_strategy("s1", {"name": "S2"})
        args, _ = mock_req.call_args
        assert args[0] == "PUT"
        assert "s1" in args[1]

    @patch("api_client.requests.request")
    def test_delete_cut_strategy(self, mock_req):
        mock_req.return_value = _mock_response(200, {"deleted": True})
        delete_cut_strategy("s1")
        args, _ = mock_req.call_args
        assert args[0] == "DELETE"
        assert "s1" in args[1]


# ---------------------------------------------------------------------------
# Mix functions
# ---------------------------------------------------------------------------

class TestMixFunctions:
    @patch("api_client.requests.request")
    def test_search_mixes(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"_id": "m1"}])
        result = search_mixes({"mfgType": "DSI"})
        args, kwargs = mock_req.call_args
        assert "mixes/search" in args[1]
        assert result == [{"_id": "m1"}]

    @patch("api_client.requests.request")
    def test_search_mix_metrics(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"_id": "m1:b1"}])
        result = search_mix_metrics({"mixId": "m1"})
        args, kwargs = mock_req.call_args
        assert "metrics/search" in args[1]
        assert kwargs["json"] == {"mixId": "m1"}
        assert result == [{"_id": "m1:b1"}]


# ---------------------------------------------------------------------------
# Job functions
# ---------------------------------------------------------------------------

class TestJobFunctions:
    @patch("api_client.requests.request")
    def test_list_jobs(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"jobId": "j1"}])
        result = list_jobs()
        args, _ = mock_req.call_args
        assert args[0] == "GET"
        assert args[1].endswith("/jobs")
        assert result == [{"jobId": "j1"}]

    @patch("api_client.requests.request")
    def test_get_job(self, mock_req):
        mock_req.return_value = _mock_response(200, {"jobId": "j1"})
        result = get_job("j1")
        args, _ = mock_req.call_args
        assert args[0] == "GET"
        assert "j1" in args[1]

    @patch("api_client.requests.request")
    def test_submit_job(self, mock_req):
        mock_req.return_value = _mock_response(201, {"jobId": "j2"})
        result = submit_job({"runId": "run1"})
        args, _ = mock_req.call_args
        assert args[0] == "POST"
        assert args[1].endswith("/jobs")

    @patch("api_client.requests.request")
    def test_cancel_job(self, mock_req):
        mock_req.return_value = _mock_response(200, {"status": "cancelled"})
        result = cancel_job("j1")
        args, _ = mock_req.call_args
        assert args[0] == "POST"
        assert "j1/cancel" in args[1]

    @patch("api_client.requests.request")
    def test_get_job_404(self, mock_req):
        mock_req.return_value = _mock_response(404, {"detail": "job not found"})
        with pytest.raises(APIError) as exc_info:
            get_job("missing")
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "job not found"


# ---------------------------------------------------------------------------
# Config functions
# ---------------------------------------------------------------------------

class TestConfigFunctions:
    @patch("api_client.requests.request")
    def test_get_all_configs(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"key": "k1"}])
        result = get_all_configs()
        args, _ = mock_req.call_args
        assert args[0] == "GET"
        assert args[1].endswith("/config")
        assert result == [{"key": "k1"}]

    @patch("api_client.requests.request")
    def test_update_config(self, mock_req):
        payload = {
            "value": 42,
            "valueType": "int",
            "description": "Test config",
            "minValue": 0,
            "maxValue": 100,
        }
        mock_req.return_value = _mock_response(200, {"key": "k1", "value": 42})
        result = update_config("k1", payload)
        args, kwargs = mock_req.call_args
        assert args[0] == "PUT"
        assert "k1" in args[1]
        assert kwargs["json"] == payload
        assert result["value"] == 42

    @patch("api_client.requests.request")
    def test_batch_update_configs(self, mock_req):
        mock_req.return_value = _mock_response(200, {"total": 2, "successful": 2})
        configs = [{
            "key": "k1",
            "update": {
                "value": 1,
                "valueType": "int",
                "description": "desc",
                "minValue": 0,
                "maxValue": 10,
            },
        }]
        result = batch_update_configs(configs)
        args, kwargs = mock_req.call_args
        assert args[0] == "POST"
        assert "batch" in args[1]
        assert kwargs["json"] == {"configs": configs, "validateOnly": False}
        assert result == {"total": 2, "successful": 2}

    @patch("api_client.requests.request")
    def test_batch_update_configs_validate_only(self, mock_req):
        mock_req.return_value = _mock_response(200, {"valid": True})
        configs = [{
            "key": "k1",
            "update": {
                "value": 1,
                "valueType": "int",
                "description": "desc",
                "minValue": 0,
                "maxValue": 10,
            },
        }]
        batch_update_configs(configs, validate_only=True)
        _, kwargs = mock_req.call_args
        assert kwargs["json"] == {"configs": configs, "validateOnly": True}
        assert "params" not in kwargs


class TestLineFunctions:
    @patch("api_client.requests.request")
    def test_list_lines(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"lineId": "DSI884"}])
        result = list_lines()
        args, _ = mock_req.call_args
        assert args[0] == "GET"
        assert args[1].endswith("/lines")
        assert result == [{"lineId": "DSI884"}]

    @patch("api_client.requests.request")
    def test_get_active_lines(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"lineId": "DSI884", "isActive": True}])
        result = get_active_lines()
        args, _ = mock_req.call_args
        assert args[0] == "GET"
        assert args[1].endswith("/lines/active")
        assert result[0]["isActive"] is True

    @patch("api_client.requests.request")
    def test_create_line(self, mock_req):
        payload = {"lineId": "DSI884", "friendlyName": "DSI 884", "lineType": "DSI884", "plant": "FSP", "hoursOfLaborAvailablePerShift": 8.0, "unitsAvailable": 2, "permittedCutStrategyIds": []}
        mock_req.return_value = _mock_response(201, payload)
        result = create_line(payload)
        args, kwargs = mock_req.call_args
        assert args[0] == "POST"
        assert args[1].endswith("/lines")
        assert kwargs["json"] == payload
        assert result["lineId"] == "DSI884"

    @patch("api_client.requests.request")
    def test_update_line(self, mock_req):
        payload = {"friendlyName": "DSI 884", "lineType": "DSI884", "plant": "FSP", "hoursOfLaborAvailablePerShift": 8.0, "unitsAvailable": 2, "isActive": True, "permittedCutStrategyIds": []}
        mock_req.return_value = _mock_response(200, {"lineId": "DSI884"})
        result = update_line("DSI884", payload)
        args, kwargs = mock_req.call_args
        assert args[0] == "PUT"
        assert args[1].endswith("/lines/DSI884")
        assert kwargs["json"] == payload
        assert result["lineId"] == "DSI884"

    @patch("api_client.requests.request")
    def test_delete_line(self, mock_req):
        mock_req.return_value = _mock_response(200, {"deleted": True})
        result = delete_line("DSI884")
        args, _ = mock_req.call_args
        assert args[0] == "DELETE"
        assert args[1].endswith("/lines/DSI884")
        assert result["deleted"] is True


class TestMonthlyContractFunctions:
    @patch("api_client.requests.request")
    def test_search_monthly_contract_demands_bulk(self, mock_req):
        mock_req.return_value = _mock_response(200, [{"skuId": "50624"}])
        result = search_monthly_contract_demands_bulk({"skuIds": ["50624"], "yearMonths": ["2026-01"]})
        args, kwargs = mock_req.call_args
        assert args[0] == "POST"
        assert "monthly-contracts/bulk-search" in args[1]
        assert kwargs["json"] == {"skuIds": ["50624"], "yearMonths": ["2026-01"]}
        assert result == [{"skuId": "50624"}]


# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------

class TestEnvConfig:
    def test_default_urls_contain_localhost(self):
        assert "localhost" in api_client.ENUMERATION_API_URL
        assert "localhost" in api_client.WORKER_API_URL
        assert "localhost" in api_client.CONFIG_API_URL

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ENUMERATION_API_URL", "http://myhost:9000/api/enumeration")
        # Reload module to pick up new env var
        importlib.reload(api_client)
        assert api_client.ENUMERATION_API_URL == "http://myhost:9000/api/enumeration"
        # Restore
        monkeypatch.delenv("ENUMERATION_API_URL", raising=False)
        importlib.reload(api_client)
