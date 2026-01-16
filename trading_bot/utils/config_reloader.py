"""
Config Reloader - Force reload configuration without restarting bot
Fixes Python caching issue where config changes require full restart
"""

import importlib
import sys
from pathlib import Path


def reload_config():
    """
    Force reload strategy_config module to pick up changes

    Returns:
        bool: True if reload successful
    """
    try:
        # Find the config module
        config_module_name = 'config.strategy_config'

        # Check if module is already loaded
        if config_module_name in sys.modules:
            # Reload the module
            config_module = sys.modules[config_module_name]
            importlib.reload(config_module)
            print("[OK] Config reloaded successfully")
            return True
        else:
            # Module not loaded yet, just import it
            import config.strategy_config
            print("[OK] Config loaded")
            return True

    except Exception as e:
        print(f"[ERROR] Failed to reload config: {e}")
        return False


def clear_pycache(directory: str = None):
    """
    Clear Python bytecode cache files

    Args:
        directory: Directory to clear (defaults to trading_bot directory)
    """
    if directory is None:
        # Get trading_bot directory
        directory = Path(__file__).parent.parent
    else:
        directory = Path(directory)

    # Find and remove all __pycache__ directories
    pycache_dirs = list(directory.rglob('__pycache__'))
    pyc_files = list(directory.rglob('*.pyc'))

    removed_count = 0

    for pycache_dir in pycache_dirs:
        try:
            import shutil
            shutil.rmtree(pycache_dir)
            removed_count += 1
        except Exception as e:
            print(f"[WARN] Could not remove {pycache_dir}: {e}")

    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            removed_count += 1
        except Exception as e:
            print(f"[WARN] Could not remove {pyc_file}: {e}")

    if removed_count > 0:
        print(f"[OK] Cleared {removed_count} cache files")
    else:
        print("ℹ️ No cache files found")


def get_current_config() -> dict:
    """
    Get current config values as a dictionary

    Returns:
        dict: Current config values
    """
    try:
        from config import strategy_config

        config_dict = {
            # Lot sizing
            'BASE_LOT_SIZE': strategy_config.BASE_LOT_SIZE,
            'USE_FIXED_LOT_SIZE': strategy_config.USE_FIXED_LOT_SIZE,

            # Grid
            'GRID_ENABLED': strategy_config.GRID_ENABLED,
            'GRID_SPACING_PIPS': strategy_config.GRID_SPACING_PIPS,
            'MAX_GRID_LEVELS': strategy_config.MAX_GRID_LEVELS,
            'GRID_LOT_SIZE': strategy_config.GRID_LOT_SIZE,

            # Hedge
            'HEDGE_ENABLED': strategy_config.HEDGE_ENABLED,
            'HEDGE_TRIGGER_PIPS': strategy_config.HEDGE_TRIGGER_PIPS,
            'HEDGE_RATIO': strategy_config.HEDGE_RATIO,
            'MAX_HEDGES_PER_POSITION': strategy_config.MAX_HEDGES_PER_POSITION,

            # DCA
            'DCA_ENABLED': strategy_config.DCA_ENABLED,
            'DCA_TRIGGER_PIPS': strategy_config.DCA_TRIGGER_PIPS,
            'DCA_MAX_LEVELS': strategy_config.DCA_MAX_LEVELS,
            'DCA_MULTIPLIER': strategy_config.DCA_MULTIPLIER,

            # Risk
            'MAX_DRAWDOWN_PERCENT': strategy_config.MAX_DRAWDOWN_PERCENT,
            'MAX_OPEN_POSITIONS': strategy_config.MAX_OPEN_POSITIONS,
            'MAX_POSITIONS_PER_SYMBOL': strategy_config.MAX_POSITIONS_PER_SYMBOL,

            # Trend Filter
            'TREND_FILTER_ENABLED': strategy_config.TREND_FILTER_ENABLED,
            'ADX_THRESHOLD': strategy_config.ADX_THRESHOLD,
            'CANDLE_LOOKBACK': strategy_config.CANDLE_LOOKBACK,
        }

        return config_dict

    except Exception as e:
        print(f"[ERROR] Failed to get config: {e}")
        return {}


def print_current_config():
    """Print current config values in a readable format"""
    config = get_current_config()

    if not config:
        print("[ERROR] Could not load config")
        return

    print()
    print("=" * 60)
    print("📋 CURRENT CONFIGURATION")
    print("=" * 60)
    print()

    print(" LOT SIZING:")
    print(f"   BASE_LOT_SIZE: {config['BASE_LOT_SIZE']}")
    print(f"   USE_FIXED_LOT_SIZE: {config['USE_FIXED_LOT_SIZE']}")
    print()

    print("🔲 GRID TRADING:")
    print(f"   ENABLED: {config['GRID_ENABLED']}")
    print(f"   SPACING: {config['GRID_SPACING_PIPS']} pips")
    print(f"   MAX_LEVELS: {config['MAX_GRID_LEVELS']}")
    print(f"   LOT_SIZE: {config['GRID_LOT_SIZE']}")
    print()

    print("🛡️ HEDGING:")
    print(f"   ENABLED: {config['HEDGE_ENABLED']}")
    print(f"   TRIGGER: {config['HEDGE_TRIGGER_PIPS']} pips")
    print(f"   RATIO: {config['HEDGE_RATIO']}x")
    print(f"   MAX_HEDGES: {config['MAX_HEDGES_PER_POSITION']}")
    print()

    print(" DCA/MARTINGALE:")
    print(f"   ENABLED: {config['DCA_ENABLED']}")
    print(f"   TRIGGER: {config['DCA_TRIGGER_PIPS']} pips")
    print(f"   MAX_LEVELS: {config['DCA_MAX_LEVELS']}")
    print(f"   MULTIPLIER: {config['DCA_MULTIPLIER']}x")
    print()

    print("[WARN] RISK MANAGEMENT:")
    print(f"   MAX_DRAWDOWN: {config['MAX_DRAWDOWN_PERCENT']}%")
    print(f"   MAX_POSITIONS: {config['MAX_OPEN_POSITIONS']}")
    print(f"   MAX_PER_SYMBOL: {config['MAX_POSITIONS_PER_SYMBOL']}")
    print()

    print(" TREND FILTER:")
    print(f"   ENABLED: {config['TREND_FILTER_ENABLED']}")
    print(f"   ADX_THRESHOLD: {config['ADX_THRESHOLD']}")
    print(f"   CANDLE_LOOKBACK: {config['CANDLE_LOOKBACK']}")
    print()
    print("=" * 60)
    print()


if __name__ == '__main__':
    """Allow running as standalone script to reload config"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("🔄 Reloading configuration...")
    reload_config()
    print_current_config()
