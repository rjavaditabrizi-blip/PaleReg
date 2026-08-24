from .base import DataProvider
from .coinbase_orderbook import CoinbaseOrderBookProvider
from .kraken_orderbook import KrakenOrderBookProvider
from .orderbook_base import OrderBookProvider
from .yfinance_provider import YFinanceProvider

_REGISTRY: dict[str, type[DataProvider]] = {
    "yfinance": YFinanceProvider,
}

_ORDERBOOK_REGISTRY: dict[str, type[OrderBookProvider]] = {
    "coinbase": CoinbaseOrderBookProvider,
    "kraken": KrakenOrderBookProvider,
}


def get_provider(name: str = "yfinance") -> DataProvider:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def register_provider(name: str, cls: type[DataProvider]) -> None:
    """Add a new data source at runtime, e.g. register_provider('polygon', PolygonProvider)."""
    _REGISTRY[name] = cls


def get_orderbook_provider(name: str = "coinbase") -> OrderBookProvider:
    if name not in _ORDERBOOK_REGISTRY:
        raise ValueError(f"Unknown order book provider '{name}'. Available: {list(_ORDERBOOK_REGISTRY)}")
    return _ORDERBOOK_REGISTRY[name]()


def register_orderbook_provider(name: str, cls: type[OrderBookProvider]) -> None:
    _ORDERBOOK_REGISTRY[name] = cls
