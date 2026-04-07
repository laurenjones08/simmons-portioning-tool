"""Business logic for production line management."""

from datetime import datetime, timezone
from typing import List, Optional

from models.line import Line, LineCreate, LineUpdate
from repositories.line_repository import LineRepository
from services.cut_strategy_catalog import CutStrategyCatalogClient


class LineService:
    """Service layer for CRUD operations on production lines."""

    def __init__(self, repository: LineRepository, cut_strategy_catalog: CutStrategyCatalogClient):
        self.repository = repository
        self.cut_strategy_catalog = cut_strategy_catalog

    def list_lines(self) -> List[Line]:
        return [Line(**document) for document in self.repository.find_all()]

    def list_active_lines(self) -> List[Line]:
        return [Line(**document) for document in self.repository.find_active()]

    def get_line(self, line_id: str) -> Optional[Line]:
        document = self.repository.find_by_id(line_id)
        if document is None:
            return None
        return Line(**document)

    def create_line(self, payload: LineCreate) -> Line:
        self._validate_cut_strategy_ids(payload.line_type.value, payload.permitted_cut_strategy_ids)
        now = datetime.now(timezone.utc)
        line = Line(
            line_id=payload.line_id,
            friendly_name=payload.friendly_name,
            line_type=payload.line_type,
            plant=payload.plant,
            permitted_cut_strategy_ids=payload.permitted_cut_strategy_ids,
            is_active=payload.is_active,
            created_at=now,
            updated_at=now,
        )
        self.repository.create(line.model_dump(by_alias=True))
        return line

    def update_line(self, line_id: str, payload: LineUpdate) -> Optional[Line]:
        existing = self.repository.find_by_id(line_id)
        if existing is None:
            return None

        self._validate_cut_strategy_ids(payload.line_type.value, payload.permitted_cut_strategy_ids)
        line = Line(
            line_id=line_id,
            friendly_name=payload.friendly_name,
            line_type=payload.line_type,
            plant=payload.plant,
            permitted_cut_strategy_ids=payload.permitted_cut_strategy_ids,
            is_active=payload.is_active,
            created_at=existing["createdAt"],
            updated_at=datetime.now(timezone.utc),
        )
        self.repository.update(line_id, line.model_dump(by_alias=True))
        return line

    def delete_line(self, line_id: str) -> bool:
        return self.repository.delete(line_id)

    def _validate_cut_strategy_ids(self, line_type: str, strategy_ids: List[str]) -> None:
        strategy_lookup = {
            str(strategy.get("_id", "")).strip(): str(
                strategy.get("lineType", strategy.get("mfgType", ""))
            ).strip()
            for strategy in self.cut_strategy_catalog.list_cut_strategies()
            if str(strategy.get("_id", "")).strip()
        }
        invalid_ids = [strategy_id for strategy_id in strategy_ids if strategy_id not in strategy_lookup]
        if invalid_ids:
            invalid_list = ", ".join(invalid_ids)
            raise ValueError(
                f"Invalid cut strategy id(s): {invalid_list}. Use ids returned by the Enumeration API."
            )
        mismatched_ids = [
            strategy_id for strategy_id in strategy_ids if strategy_lookup.get(strategy_id) != line_type
        ]
        if mismatched_ids:
            mismatch_list = ", ".join(mismatched_ids)
            raise ValueError(
                f"Cut strategy id(s) {mismatch_list} do not match lineType '{line_type}'."
            )
