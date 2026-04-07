"""Enumeration API client used for cut strategy validation."""

import json
from typing import List
from urllib import error, request


class CutStrategyCatalogClient:
    """Fetch cut strategies from the Enumeration API."""

    def __init__(self, enumeration_api_url: str, timeout_seconds: float = 5.0):
        self.enumeration_api_url = enumeration_api_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_cut_strategies(self) -> List[dict]:
        payload = json.dumps({}).encode("utf-8")
        req = request.Request(
            f"{self.enumeration_api_url}/cut-strategies/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise Exception(
                f"Enumeration API returned HTTP {exc.code} while loading cut strategies: {detail}"
            ) from exc
        except error.URLError as exc:
            raise Exception(f"Could not reach Enumeration API: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise Exception("Enumeration API returned invalid JSON for cut strategies") from exc
