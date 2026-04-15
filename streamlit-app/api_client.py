"""
API client for the Streamlit Settings Page.

Reads base URLs from environment variables with localhost defaults.
All public functions raise APIError on non-2xx responses.
"""

import os
import requests

# ---------------------------------------------------------------------------
# Base URL configuration
# ---------------------------------------------------------------------------

ENUMERATION_API_URL = os.getenv(
    "ENUMERATION_API_URL", "http://localhost:8080/api/enumeration"
)
WORKER_API_URL = os.getenv(
    "WORKER_API_URL", "http://localhost:8080/api/enumeration-worker"
)
SCHEDULING_WORKER_API_URL = os.getenv(
    "SCHEDULING_WORKER_API_URL", "http://localhost:8080/api/scheduling-worker"
)
CONFIG_API_URL = os.getenv(
    "CONFIG_API_URL", "http://localhost:8080/api/config"
)
SCHEDULING_API_URL = os.getenv(
    "SCHEDULING_API_URL", "http://localhost:8080/api/scheduling"
)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Raised when an API call returns a non-2xx response or a network error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _handle_response(response: requests.Response) -> requests.Response:
    """Raise APIError if the response status is not 2xx."""
    if not response.ok:
        try:
            body = response.json()
            detail = body.get("detail", response.text)
            if not isinstance(detail, str):
                detail = str(detail)
        except Exception:
            detail = response.text or f"HTTP error {response.status_code}"
        raise APIError(status_code=response.status_code, detail=detail)
    return response


def _request(method: str, url: str, **kwargs) -> requests.Response:
    """Execute an HTTP request, wrapping network errors in APIError."""
    try:
        response = requests.request(method, url, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise APIError(status_code=0, detail=f"Network error: {exc}") from exc
    return _handle_response(response)


# ---------------------------------------------------------------------------
# Buckets  (Enumeration API)
# ---------------------------------------------------------------------------

def search_buckets(criteria: dict) -> list[dict]:
    resp = _request("POST", f"{ENUMERATION_API_URL}/buckets/search", json=criteria)
    return resp.json()


def create_bucket(payload: dict) -> dict:
    resp = _request("POST", f"{ENUMERATION_API_URL}/buckets", json=payload)
    return resp.json()


def update_bucket(bucket_id: str, payload: dict) -> dict:
    resp = _request("PUT", f"{ENUMERATION_API_URL}/buckets/{bucket_id}", json=payload)
    return resp.json()


def delete_bucket(bucket_id: str) -> dict:
    resp = _request("DELETE", f"{ENUMERATION_API_URL}/buckets/{bucket_id}")
    return resp.json()


# ---------------------------------------------------------------------------
# SKUs  (Enumeration API)
# ---------------------------------------------------------------------------

def search_skus(criteria: dict) -> list[dict]:
    resp = _request("POST", f"{ENUMERATION_API_URL}/skus/search", json=criteria)
    return resp.json()


def create_sku(payload: dict) -> dict:
    resp = _request("POST", f"{ENUMERATION_API_URL}/skus", json=payload)
    return resp.json()


def update_sku(sku_id: str, payload: dict) -> dict:
    resp = _request("PUT", f"{ENUMERATION_API_URL}/skus/{sku_id}", json=payload)
    return resp.json()


def delete_sku(sku_id: str) -> dict:
    resp = _request("DELETE", f"{ENUMERATION_API_URL}/skus/{sku_id}")
    return resp.json()


def batch_import_skus(skus: list[dict]) -> dict:
    payload = {"skus": skus, "validateOnly": False}
    resp = _request("POST", f"{ENUMERATION_API_URL}/skus/batch", json=payload)
    return resp.json()


# ---------------------------------------------------------------------------
# Cut Strategies  (Enumeration API)
# ---------------------------------------------------------------------------

def search_cut_strategies(criteria: dict) -> list[dict]:
    resp = _request("POST", f"{ENUMERATION_API_URL}/cut-strategies/search", json=criteria)
    return resp.json()


def create_cut_strategy(payload: dict) -> dict:
    resp = _request("POST", f"{ENUMERATION_API_URL}/cut-strategies", json=payload)
    return resp.json()


def update_cut_strategy(strategy_id: str, payload: dict) -> dict:
    resp = _request(
        "PUT", f"{ENUMERATION_API_URL}/cut-strategies/{strategy_id}", json=payload
    )
    return resp.json()


def delete_cut_strategy(strategy_id: str) -> dict:
    resp = _request("DELETE", f"{ENUMERATION_API_URL}/cut-strategies/{strategy_id}")
    return resp.json()


# ---------------------------------------------------------------------------
# Mixes  (Enumeration API)
# ---------------------------------------------------------------------------

def search_mixes(criteria: dict) -> list[dict]:
    resp = _request("POST", f"{ENUMERATION_API_URL}/mixes/search", json=criteria)
    return resp.json()


# ---------------------------------------------------------------------------
# Mix Metrics  (Enumeration API)
# ---------------------------------------------------------------------------

def search_mix_metrics(criteria: dict) -> list[dict]:
    resp = _request("POST", f"{ENUMERATION_API_URL}/metrics/search", json=criteria)
    return resp.json()


# ---------------------------------------------------------------------------
# Jobs  (Worker API)
# ---------------------------------------------------------------------------

def list_jobs() -> list[dict]:
    resp = _request("GET", f"{WORKER_API_URL}/jobs")
    return resp.json()


def get_job(job_id: str) -> dict:
    resp = _request("GET", f"{WORKER_API_URL}/jobs/{job_id}")
    return resp.json()


def submit_job(payload: dict) -> dict:
    resp = _request("POST", f"{WORKER_API_URL}/jobs", json=payload)
    return resp.json()


def cancel_job(job_id: str) -> dict:
    resp = _request("POST", f"{WORKER_API_URL}/jobs/{job_id}/cancel")
    return resp.json()


def list_scheduling_jobs(status_filter: str | None = None) -> list[dict]:
    params = {"status": status_filter} if status_filter else None
    resp = _request("GET", f"{SCHEDULING_WORKER_API_URL}/jobs", params=params)
    return resp.json()


def get_scheduling_job(job_id: str) -> dict:
    resp = _request("GET", f"{SCHEDULING_WORKER_API_URL}/jobs/{job_id}")
    return resp.json()


def submit_scheduling_job(payload: dict) -> dict:
    resp = _request("POST", f"{SCHEDULING_WORKER_API_URL}/jobs", json=payload)
    return resp.json()


def cancel_scheduling_job(job_id: str) -> dict:
    resp = _request("POST", f"{SCHEDULING_WORKER_API_URL}/jobs/{job_id}/cancel")
    return resp.json()


def get_scheduling_job_artifacts(job_id: str) -> list[dict]:
    resp = _request("GET", f"{SCHEDULING_API_URL}/jobs/{job_id}/artifacts")
    return resp.json()


# ---------------------------------------------------------------------------
# Config  (Config API)
# ---------------------------------------------------------------------------

def get_all_configs() -> list[dict]:
    resp = _request("GET", f"{CONFIG_API_URL}/config")
    return resp.json()


def update_config(key: str, payload: dict) -> dict:
    resp = _request("PUT", f"{CONFIG_API_URL}/config/{key}", json=payload)
    return resp.json()


def batch_update_configs(configs: list[dict], validate_only: bool = False) -> dict:
    payload = {
        "configs": configs,
        "validateOnly": validate_only,
    }
    resp = _request("POST", f"{CONFIG_API_URL}/config/batch", json=payload)
    return resp.json()


# ---------------------------------------------------------------------------
# Lines  (Config API)
# ---------------------------------------------------------------------------

def list_lines() -> list[dict]:
    resp = _request("GET", f"{CONFIG_API_URL}/lines")
    return resp.json()


def get_active_lines() -> list[dict]:
    resp = _request("GET", f"{CONFIG_API_URL}/lines/active")
    return resp.json()


def create_line(payload: dict) -> dict:
    resp = _request("POST", f"{CONFIG_API_URL}/lines", json=payload)
    return resp.json()


def update_line(line_id: str, payload: dict) -> dict:
    resp = _request("PUT", f"{CONFIG_API_URL}/lines/{line_id}", json=payload)
    return resp.json()


def delete_line(line_id: str) -> dict:
    resp = _request("DELETE", f"{CONFIG_API_URL}/lines/{line_id}")
    return resp.json()


# ---------------------------------------------------------------------------
# Available WIP  (Scheduling API)
# ---------------------------------------------------------------------------

def search_available_wip(criteria: dict) -> list[dict]:
    resp = _request("POST", f"{SCHEDULING_API_URL}/available-wip/search", json=criteria)
    return resp.json()


def create_available_wip(payload: dict) -> dict:
    resp = _request("POST", f"{SCHEDULING_API_URL}/available-wip", json=payload)
    return resp.json()


def update_available_wip(wip_id: str, payload: dict) -> dict:
    resp = _request("PUT", f"{SCHEDULING_API_URL}/available-wip/{wip_id}", json=payload)
    return resp.json()


def delete_available_wip(wip_id: str) -> dict:
    resp = _request("DELETE", f"{SCHEDULING_API_URL}/available-wip/{wip_id}")
    return resp.json()


# ---------------------------------------------------------------------------
# SKU Demands  (Scheduling API)
# ---------------------------------------------------------------------------

def search_sku_demands(criteria: dict) -> list[dict]:
    resp = _request("POST", f"{SCHEDULING_API_URL}/sku-demands/search", json=criteria)
    return resp.json()


def create_sku_demand(payload: dict) -> dict:
    resp = _request("POST", f"{SCHEDULING_API_URL}/sku-demands", json=payload)
    return resp.json()


def bulk_create_sku_demands(payload: dict) -> dict:
    resp = _request("POST", f"{SCHEDULING_API_URL}/sku-demands/bulk", json=payload)
    return resp.json()


def update_sku_demand(demand_id: str, payload: dict) -> dict:
    resp = _request("PUT", f"{SCHEDULING_API_URL}/sku-demands/{demand_id}", json=payload)
    return resp.json()


def delete_sku_demand(demand_id: str) -> dict:
    resp = _request("DELETE", f"{SCHEDULING_API_URL}/sku-demands/{demand_id}")
    return resp.json()


# ---------------------------------------------------------------------------
# Monthly Contract Demands  (Scheduling API)
# ---------------------------------------------------------------------------

def search_monthly_contract_demands(criteria: dict) -> list[dict]:
    resp = _request(
        "POST",
        f"{SCHEDULING_API_URL}/monthly-contracts/search",
        json=criteria,
    )
    return resp.json()


def create_monthly_contract_demand(payload: dict) -> dict:
    resp = _request("POST", f"{SCHEDULING_API_URL}/monthly-contracts", json=payload)
    return resp.json()


def bulk_create_monthly_contract_demands(payload: dict) -> dict:
    resp = _request("POST", f"{SCHEDULING_API_URL}/monthly-contracts/bulk", json=payload)
    return resp.json()


def update_monthly_contract_demand(contract_id: str, payload: dict) -> dict:
    resp = _request(
        "PUT",
        f"{SCHEDULING_API_URL}/monthly-contracts/{contract_id}",
        json=payload,
    )
    return resp.json()


def delete_monthly_contract_demand(contract_id: str) -> dict:
    resp = _request("DELETE", f"{SCHEDULING_API_URL}/monthly-contracts/{contract_id}")
    return resp.json()
