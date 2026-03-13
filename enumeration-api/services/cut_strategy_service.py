"""Business logic layer for CutStrategy operations."""

from typing import Dict, List, Optional

from pymongo.errors import DuplicateKeyError

from models.cut_strategy import (
    CutStrategy,
    CutStrategyCreate,
    CutStrategySearchCriteria,
    CutStrategyUpdate,
)
from repositories.cut_strategy_repository import CutStrategyRepository
from repositories.mix_metric_repository import MixMetricRepository
from repositories.mix_repository import MixRepository


class CutStrategyService:
    def __init__(
        self,
        repository: CutStrategyRepository,
        mix_repository: MixRepository,
        mix_metric_repository: MixMetricRepository,
    ):
        self.repository = repository
        self.mix_repository = mix_repository
        self.mix_metric_repository = mix_metric_repository

    def create_cut_strategy(self, payload: CutStrategyCreate) -> CutStrategy:
        strategy = CutStrategy(**payload.model_dump(by_alias=True))
        document = strategy.model_dump(by_alias=True)

        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A cut strategy with this name already exists for the mfgType")

        return CutStrategy(**inserted)

    def get_cut_strategy_by_id(self, strategy_id: str) -> Optional[CutStrategy]:
        strategy_doc = self.repository.get_by_id(strategy_id)
        return CutStrategy(**strategy_doc) if strategy_doc else None

    def search_cut_strategies(self, criteria: CutStrategySearchCriteria) -> List[CutStrategy]:
        raw = criteria.model_dump(by_alias=True, exclude_none=True)
        mongo_criteria = {k: v for k, v in raw.items() if k != "includesPart"}

        includes_part = raw.get("includesPart")
        if includes_part:
            mongo_criteria["parts"] = includes_part

        docs = self.repository.search(mongo_criteria)
        return [CutStrategy(**doc) for doc in docs]

    def update_cut_strategy(self, strategy_id: str, payload: CutStrategyUpdate) -> Optional[CutStrategy]:
        strategy = CutStrategy(_id=strategy_id, **payload.model_dump(by_alias=True))
        document = strategy.model_dump(by_alias=True)

        try:
            updated = self.repository.update(strategy_id, document)
        except DuplicateKeyError:
            raise ValueError("A cut strategy with this name already exists for the mfgType")

        if not updated:
            return None
        return strategy

    def delete_cut_strategy(self, strategy_id: str) -> Dict[str, int | bool]:
        mix_ids = self.mix_repository.get_ids_by_cut_strategy_id(strategy_id)
        metrics_deleted = self.mix_metric_repository.delete_by_mix_ids(mix_ids)
        mixes_deleted = self.mix_repository.delete_by_cut_strategy_id(strategy_id)
        deleted = self.repository.delete(strategy_id)

        return {
            "deleted": deleted,
            "mixes_deleted": mixes_deleted,
            "metrics_deleted": metrics_deleted,
        }
