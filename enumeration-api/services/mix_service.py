"""Business logic layer for MIX operations."""

from typing import List, Optional, Dict

from pymongo.errors import DuplicateKeyError

from models.mix import MIX, MixCreate, MixSearchCriteria, MixUpdate
from repositories.mix_repository import MixRepository


class MixService:
    def __init__(self, repository: MixRepository):
        self.repository = repository

    @staticmethod
    def _build_sku_set_key(skus: Dict[str, str]) -> str:
        # Uniqueness is based on SKU trade-number set, independent of part IDs.
        return "|".join(sorted(str(k).strip() for k in skus.keys()))

    def create_mix(self, payload: MixCreate) -> MIX:
        mix = MIX(**payload.model_dump(by_alias=True))
        document = mix.model_dump(by_alias=True)
        document["skuSetKey"] = self._build_sku_set_key(mix.skus)

        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A mix already exists for this SKU set and mfgType")

        return MIX(**inserted)

    def get_mix_by_id(self, mix_id: str) -> Optional[MIX]:
        mix_doc = self.repository.get_by_id(mix_id)
        return MIX(**mix_doc) if mix_doc else None

    def search_mixes(self, criteria: MixSearchCriteria) -> List[MIX]:
        raw = criteria.model_dump(by_alias=True, exclude_none=True)
        mongo_criteria = {k: v for k, v in raw.items() if k != "skuTradeNumber"}

        sku_trade_number = raw.get("skuTradeNumber")
        if sku_trade_number:
            mongo_criteria[f"skus.{sku_trade_number}"] = {"$exists": True}

        docs = self.repository.search(mongo_criteria)
        return [MIX(**doc) for doc in docs]

    def update_mix(self, mix_id: str, payload: MixUpdate) -> Optional[MIX]:
        mix = MIX(_id=mix_id, **payload.model_dump(by_alias=True))
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
