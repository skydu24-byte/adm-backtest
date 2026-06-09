"""
main.py — CLI entry point for the ADM backtester.

Usage:
    python main.py                           # run with default config
    python main.py --config my_config.py     # import custom config from file
    python main.py --params key=val ...      # override individual params
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

from config import ADMConfig
from backtest import run


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_overrides(overrides: list) -> dict:
    """Parse 'key=value' pairs from CLI into a dict."""
    result = {}
    for pair in overrides:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Use 'key=value'.")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        # Type coercion: bool, int, float, list
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.startswith("[") and val.endswith("]"):
            # e.g. "[1, 3, 12]"
            import json
            val = json.loads(val)
        elif val.startswith("'") and val.endswith("'"):
            val = val.strip("'")
        elif val.startswith('"') and val.endswith('"'):
            val = val.strip('"')
        else:
            for t in (int, float):
                try:
                    val = t(val)
                    break
                except (ValueError, TypeError):
                    continue
        result[key] = val
    return result


def _config_from_file(path: str) -> type:
    """Dynamically import a config file and return its module."""
    spec = importlib.util.spec_from_file_location("user_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Accelerating Dual Momentum (ADM) Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to a Python file exporting a `cfg` ADMConfig instance.",
    )
    parser.add_argument(
        "--params", "-p",
        type=str,
        nargs="+",
        default=[],
        help="Override individual parameters, e.g. -p transaction_cost=0.002 momentum_windows_months='[1,3,12]'",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # ── Build config ───────────────────────────────────────────────────────
    if args.config:
        mod = _config_from_file(args.config)
        if not hasattr(mod, "cfg"):
            logger.error("Config file %s must define a `cfg` variable.", args.config)
            sys.exit(1)
        cfg = mod.cfg
        if not isinstance(cfg, ADMConfig):
            logger.error("`cfg` in %s must be an ADMConfig instance.", args.config)
            sys.exit(1)
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = ADMConfig()

    # Apply overrides
    if args.params:
        overrides = _parse_overrides(args.params)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                logger.info("Override: %s = %r", k, v)
            else:
                logger.warning("Unknown config field '%s' — ignored.", k)

    # ── Run ────────────────────────────────────────────────────────────────
    try:
        portfolio_df, metrics = run(cfg)
        logger.info("Backtest complete. Results in '%s/'.", cfg.output_dir)
    except Exception:
        logger.exception("Backtest failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
