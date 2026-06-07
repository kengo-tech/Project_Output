import time
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.binance.com"


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _to_milliseconds(value: str | pd.Timestamp | None) -> int | None:
    """
    Convert a date-like value to milliseconds.

    Parameters
    ----------
    value : str | pd.Timestamp | None
        Example:
        - "2026-05-01"
        - "2026-05-01 00:00:00"
        - pd.Timestamp("2026-05-01", tz="UTC")

    Returns
    -------
    int | None
        Unix timestamp in milliseconds.
    """

    if value is None:
        return None

    timestamp = pd.to_datetime(value, utc=True)

    return int(timestamp.timestamp() * 1000)


def _request_binance_klines(
    symbol: str,
    interval: str,
    limit: int = 1000,
    start_ms: int | None = None,
    end_ms: int | None = None,
    max_retries: int = 3,
    sleep_seconds: float = 0.2,
) -> list[list[Any]]:
    """
    Request kline data from Binance Spot API.

    This function only requests one page.
    Pagination is handled by get_historical_klines().
    """

    url = f"{BASE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    if start_ms is not None:
        params["startTime"] = start_ms

    if end_ms is not None:
        params["endTime"] = end_ms

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:
            last_error = error

            if attempt < max_retries - 1:
                wait_time = sleep_seconds * (attempt + 1)
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Binance API request failed after {max_retries} attempts. "
                    f"Last error: {last_error}"
                ) from error

    return []


def _clean_kline_dataframe(raw_data: list[list[Any]]) -> pd.DataFrame:
    """
    Convert raw Binance kline response to a cleaned DataFrame.
    """

    if not raw_data:
        return pd.DataFrame(columns=[col for col in KLINE_COLUMNS if col != "ignore"])

    df = pd.DataFrame(raw_data, columns=KLINE_COLUMNS)

    ### Staging / Cleaning
    clean_df = (
        df
        .copy()
        .assign(
            open_time=lambda x: pd.to_datetime(x["open_time"], unit="ms", utc=True),
            close_time=lambda x: pd.to_datetime(x["close_time"], unit="ms", utc=True),
            open=lambda x: pd.to_numeric(x["open"], errors="coerce"),
            high=lambda x: pd.to_numeric(x["high"], errors="coerce"),
            low=lambda x: pd.to_numeric(x["low"], errors="coerce"),
            close=lambda x: pd.to_numeric(x["close"], errors="coerce"),
            volume=lambda x: pd.to_numeric(x["volume"], errors="coerce"),
            quote_asset_volume=lambda x: pd.to_numeric(
                x["quote_asset_volume"],
                errors="coerce",
            ),
            number_of_trades=lambda x: pd.to_numeric(
                x["number_of_trades"],
                errors="coerce",
            ),
            taker_buy_base_volume=lambda x: pd.to_numeric(
                x["taker_buy_base_volume"],
                errors="coerce",
            ),
            taker_buy_quote_volume=lambda x: pd.to_numeric(
                x["taker_buy_quote_volume"],
                errors="coerce",
            ),
        )
        .drop(columns=["ignore"])
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    return clean_df


def get_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Download the latest Binance spot kline/candlestick data.

    This keeps backward compatibility with our existing main.py.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. BTCUSDT.
    interval : str
        Kline interval, e.g. 1m, 5m, 15m, 1h.
    limit : int
        Number of rows to download. Binance spot kline max is usually 1000.

    Returns
    -------
    pd.DataFrame
        Cleaned kline dataframe.
    """

    raw_data = _request_binance_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )

    return _clean_kline_dataframe(raw_data)


def get_historical_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    limit: int = 1000,
    sleep_seconds: float = 0.2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download historical Binance spot kline/candlestick data with pagination.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. BTCUSDT.

    interval : str
        Kline interval, e.g. 1m, 5m, 15m, 1h.

    start_time : str | pd.Timestamp | None
        Start date/time.
        Example: "2026-05-01"

    end_time : str | pd.Timestamp | None
        End date/time.
        Example: "2026-06-07"

    limit : int
        Number of rows per API request.
        Binance spot kline max is usually 1000.

    sleep_seconds : float
        Sleep between requests to be gentle with API limits.

    verbose : bool
        If True, print download progress.

    Returns
    -------
    pd.DataFrame
        Cleaned historical kline dataframe.

    Notes
    -----
    This function is for research/backtesting data collection.
    It does not place any orders.
    """

    start_ms = _to_milliseconds(start_time)
    end_ms = _to_milliseconds(end_time)

    if start_ms is None:
        raise ValueError("start_time must be specified for historical download.")

    if end_ms is not None and start_ms >= end_ms:
        raise ValueError("start_time must be earlier than end_time.")

    all_raw_data = []
    current_start_ms = start_ms

    page = 1

    while True:
        raw_data = _request_binance_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_ms=current_start_ms,
            end_ms=end_ms,
            sleep_seconds=sleep_seconds,
        )

        if not raw_data:
            if verbose:
                print("No more data returned from Binance.")
            break

        all_raw_data.extend(raw_data)

        last_open_time_ms = int(raw_data[-1][0])
        last_open_time = pd.to_datetime(last_open_time_ms, unit="ms", utc=True)

        if verbose:
            print(
                f"Page {page}: downloaded {len(raw_data)} rows. "
                f"Last open time: {last_open_time}"
            )

        page += 1

        if len(raw_data) < limit:
            break

        next_start_ms = last_open_time_ms + 1

        if end_ms is not None and next_start_ms >= end_ms:
            break

        if next_start_ms <= current_start_ms:
            raise RuntimeError(
                "Pagination did not advance. "
                "Stopping to avoid an infinite loop."
            )

        current_start_ms = next_start_ms

        time.sleep(sleep_seconds)

    clean_df = _clean_kline_dataframe(all_raw_data)

    if end_ms is not None:
        end_timestamp = pd.to_datetime(end_ms, unit="ms", utc=True)
        clean_df = (
            clean_df
            .loc[lambda x: x["open_time"] <= end_timestamp]
            .copy()
            .reset_index(drop=True)
        )

    if verbose:
        print("\nHistorical download completed.")
        print(f"Rows: {len(clean_df)}")

        if not clean_df.empty:
            print(f"Start: {clean_df['open_time'].min()}")
            print(f"End:   {clean_df['open_time'].max()}")

    return clean_df