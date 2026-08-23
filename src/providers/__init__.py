from .base import DataProvider
from .yfinance_provider import YFinanceProvider

_REGISTRY: dict[str, type[DataProvider]] = {
    "yfinance": YFinanceProvider,
}


def get_provider(name: str = "yfinance") -> DataProvider:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def register_provider(name: str, cls: type[DataProvider]) -> None:
    """Add a new data source at runtime, e.g. register_provider('polygon', PolygonProvider)."""
    _REGISTRY[name] = cls
