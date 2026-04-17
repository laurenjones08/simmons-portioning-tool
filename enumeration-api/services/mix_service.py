"""Business logic layer for MIX operations."""

from typing import List, Optional, Dict, Iterable
import logging

from bson import ObjectId
from pydantic import ValidationError
from pymongo.errors import DuplicateKeyError

from models.cut_strategy import CutStrategy
from models.mix import MIX, MixCreate, MixSearchCriteria, MixUpdate
from repositories.cut_strategy_repository import CutStrategyRepository
from repositories.mix_repository import MixRepository


logger = logging.getLogger(__name__)


class MixService:
    def __init__(self, repository: MixRepository, cut_strategy_repository: CutStrategyRepository):
        self.repository = repository
        self.cut_strategy_repository = cut_strategy_repository

    @staticmethod
    def _normalize_part_codes(parts: Iterable[str]) -> list[str]:
        return [str(part).strip().upper() for part in parts]

    @staticmethod
    def _build_sku_set_key(skus: Dict[str, str]) -> str:
        # Uniqueness is based on SKU trade-number set, independent of part IDs.
        return "|".join(sorted(str(k).strip() for k in skus.keys()))

    def _get_strategy(self, strategy_id: str) -> CutStrategy:
        strategy_doc = self.cut_strategy_repository.get_by_id(strategy_id)
        if strategy_doc is None:
            raise ValueError(f"Cut strategy with id {strategy_id} not found")

        return CutStrategy(**strategy_doc)

    def _validate_strategy_assignment(self, mix: MIX) -> None:
        strategy = self._get_strategy(mix.cut_strategy_id)

        required_parts = self._normalize_part_codes(strategy.parts)
        assigned_parts = self._normalize_part_codes(mix.skus.values())

        if mix.mfg_type != strategy.line_type:
            raise ValueError(
                "Mix mfgType must match the selected cut strategy lineType"
            )

        if set(assigned_parts) != set(required_parts):
            missing_parts = [part for part in required_parts if part not in assigned_parts]
            extra_parts = [part for part in assigned_parts if part not in required_parts]
            parts_message = []
            if missing_parts:
                parts_message.append(f"missing parts: {', '.join(missing_parts)}")
            if extra_parts:
                parts_message.append(f"unexpected parts: {', '.join(extra_parts)}")
            raise ValueError(
                "Selected cut strategy must allocate a SKU to every required part"
                + (f" ({'; '.join(parts_message)})" if parts_message else "")
            )

        if mix.num_fillets != len(required_parts):
            raise ValueError(
                f"numFillets must equal the number of parts in the selected cut strategy ({len(required_parts)})"
            )

    def create_mix(self, payload: MixCreate) -> MIX:
        mix = MIX(**payload.model_dump(by_alias=True))
        self._validate_strategy_assignment(mix)
        document = mix.model_dump(by_alias=True)
        document["skuSetKey"] = self._build_sku_set_key(mix.skus)

        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A mix already exists for this SKU set and mfgType")

        return MIX(**inserted)

    def get_mix_by_id(self, mix_id: str) -> Optional[MIX]:
        mix_doc = self.repository.get_by_id(mix_id)
        return self._safe_parse_mix(mix_doc, f"get_by_id({mix_id})")

    @staticmethod
    def _normalize_mix_doc(doc: dict) -> dict:
        """Normalize legacy Mongo types to the string fields expected by MIX."""
        normalized = dict(doc)
        for key in ("_id", "cutStrategyID", "skuSetKey"):
            value = normalized.get(key)
            if isinstance(value, ObjectId):
                normalized[key] = str(value)
        return normalized

    @classmethod
    def _safe_parse_mix(cls, doc: Optional[dict], context: str) -> Optional[MIX]:
        """Parse a raw mix document, skipping invalid legacy records."""
        if doc is None:
            return None
        try:
            return MIX(**cls._normalize_mix_doc(doc))
        except ValidationError as exc:
            logger.warning("Skipping invalid mix document during %s: %s", context, exc)
            return None

    def search_mixes(self, criteria: MixSearchCriteria) -> List[MIX]:
        raw = criteria.model_dump(by_alias=True, exclude_none=True)
        mongo_criteria = {k: v for k, v in raw.items() if k != "skuTradeNumber"}

        sku_trade_number = raw.get("skuTradeNumber")
        if sku_trade_number:
            # Query denormalized skuKeys for efficient membership lookup.
            mongo_criteria["skuKeys"] = sku_trade_number

        docs = self.repository.search(mongo_criteria)
        parsed: List[MIX] = []
        for doc in docs:
            mix = self._safe_parse_mix(doc, "search")
            if mix is not None:
                parsed.append(mix)
        return parsed

    def update_mix(self, mix_id: str, payload: MixUpdate) -> Optional[MIX]:
        mix = MIX(_id=mix_id, **payload.model_dump(by_alias=True))
        self._validate_strategy_assignment(mix)
        document = mix.model_dump(by_alias=True)
        document["skuSetKey"] = self._build_sku_set_key(mix.skus)

        try:
            updated = self.repository.update(mix_id, document)
        except DuplicateKeyError:
            raise ValueError("A mix already exists for this SKU set and mfgType")

        if not updated:
            return None
        return mix

    def delete_mix(self, mix_id: str) -> bool:
        return self.repository.delete(mix_id)
