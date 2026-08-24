"""Provider interface for fund research data."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from ..models import FundDataset


class FundDataSource(Protocol):
    def fetch(self, fund_code: str, as_of: date | None = None) -> FundDataset:
        """Fetch public data available on or before ``as_of``."""
