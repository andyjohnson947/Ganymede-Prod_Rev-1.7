# Python Trading Bot - Based on EA Analysis

## Strategy Overview

This trading bot implements the strategy discovered through reverse engineering of the EA, using:
- **Multi-timeframe analysis** (H1, D1, W1)
- **Volume Profile** (POC, VAH, VAL, HVN, LVN)
- **VWAP with deviation bands** (±1σ, ±2σ, ±3σ)
- **Confluence-based entries** (minimum score: 4)
- **Advanced recovery mechanisms** (Grid + Hedge + DCA)

## Discovered Strategy Parameters

### Entry Signals (from 428 trades analyzed)
- **VWAP Band 1 (±1σ)**: 28.0% of trades (120 times)
- **VWAP Band 2 (±2σ)**: 39.5% of trades (169 times)
- **POC (Point of Control)**: 38.1% of trades (163 times)
- **Swing Low**: 17.1% of trades (73 times)
- **Above VAH**: 14.0% of trades (60 times)
- **Below VAL**: 18.7% of trades (80 times)
- **Low Volume Node**: 16.1% of trades (69 times)

### Confluence Requirements
- **Minimum confluence score**: 4
- **Optimal score win rate**: 83.3%
- **High-value setups**: 3+ confluence factors

### Key Confluence Factors (Most Important)
1. **Prev Day VAH** - 364 occurrences
2. **Weekly HVN** - 328 occurrences
3. **Prev Day POC** - 325 occurrences
4. **Prev Week Swing Low** - 325 occurrences
5. **Daily HVN** - 310 occurrences

### Performance Metrics
- **Win Rate**: 64.3%
- **Total Trades**: 428
- **Primary Timeframe**: H1

### Recovery Mechanisms

#### Grid Trading
- **Spacing**: 10.8 pips
- **Max Levels**: 6
- **Lot Size**: 0.02 per level

#### Hedging
- **Trigger**: 8 pips underwater
- **Ratio**: 2.4x (overhedge)
- **Direction**: Opposite to original trade

#### DCA/Martingale
- **Averaging down** losing positions
- **Maximum exposure**: 5.04 lots

### Risk Management
- **Risk per trade**: 1%
- **Max total exposure**: 5.04 lots
- **Stop trading**: At 10% drawdown

## Project Structure

```
trading_bot/
├── core/
│   └── mt5_manager.py          # MT5 connection and order management
├── strategies/
│   ├── confluence_strategy.py  # Main strategy orchestration
│   ├── signal_detector.py      # Entry signal detection
│   └── recovery_manager.py     # Grid/Hedge/DCA implementation
├── indicators/
│   ├── vwap.py                 # VWAP calculation with bands
│   ├── volume_profile.py       # POC, VAH, VAL, HVN, LVN
│   └── htf_levels.py           # Higher timeframe level detection
├── utils/
│   ├── logger.py               # Logging system
│   └── risk_calculator.py      # Position sizing and risk management
├── config/
│   └── strategy_config.py      # All discovered parameters
├── gui/
│   └── trading_gui.py          # Professional GUI interface
└── main.py                     # Main bot entry point
```

## Installation

### Requirements
- Python 3.8+
- MetaTrader 5 terminal installed
- MetaTrader 5 account

### Install Dependencies

```bash
pip install MetaTrader5 pandas numpy
```

### Optional (for GUI)
```bash
pip install tk matplotlib
```

## Usage

### Command Line Mode

```bash
# Basic usage
python main.py --login 12345 --password "yourpass" --server "Broker-Server" --symbols EURUSD GBPUSD

# With custom symbols
python main.py --login 12345 --password "yourpass" --server "Broker-Server" --symbols EURUSD GBPUSD USDJPY
```

### GUI Mode

```bash
python main.py --gui
```

The GUI provides:
- **Real-time monitoring**: Account balance, equity, margin, P/L
- **Position tracking**: All open positions with recovery status
- **Risk metrics**: Drawdown, exposure, position count
- **Statistics**: Trades, signals, grid levels, hedges, DCA
- **Activity log**: Real-time bot activity
- **Controls**: Start/stop trading, connect/disconnect MT5

## Configuration

Edit `config/strategy_config.py` to customize:

- **Symbols**: Trading pairs
- **Timeframes**: Analysis timeframes
- **Confluence weights**: Importance of each factor
- **Grid parameters**: Spacing, max levels, lot size
- **Hedge parameters**: Trigger pips, ratio
- **DCA parameters**: Trigger, multiplier
- **Risk limits**: Risk %, max exposure, max drawdown

## Features

### Signal Detection
- ✅ Multi-factor confluence scoring
- ✅ VWAP band analysis
- ✅ Volume profile integration
- ✅ HTF level detection
- ✅ Weighted scoring (HTF = higher weights)
- ✅ Session and day filtering

### Position Management
- ✅ Automatic entry on high-confluence signals
- ✅ VWAP reversion exits
- ✅ Grid trading (10.8 pip spacing)
- ✅ Hedging (8 pips, 2.4x ratio)
- ✅ DCA/Martingale (1.5x multiplier)
- ✅ Position tracking and monitoring

### Risk Management
- ✅ Position sizing (1% risk per trade)
- ✅ Total exposure limits (5.04 lots max)
- ✅ Drawdown monitoring (10% max)
- ✅ Margin level checking
- ✅ Free margin validation

### Monitoring & Logging
- ✅ Real-time GUI interface
- ✅ File logging (main, trades, signals)
- ✅ Console output
- ✅ Statistics tracking
- ✅ Performance metrics

## Important Notes

### ⚠️ Risk Warning
- This bot uses aggressive recovery mechanisms (Grid + Hedge + DCA)
- Maximum exposure can reach 5.04 lots
- Only use with proper risk management
- Test thoroughly on demo account first
- Never risk more than you can afford to lose

### 🔧 Customization
- All parameters in `config/strategy_config.py` can be adjusted
- Start with conservative settings on demo
- Monitor performance and adjust gradually
- Keep detailed logs of all changes

### 📊 Monitoring
- Check GUI regularly for position status
- Monitor drawdown levels
- Track recovery mechanism usage
- Review log files for errors

## Development Roadmap

### Completed ✅
- [x] Core MT5 integration
- [x] VWAP indicator
- [x] Volume Profile indicator
- [x] HTF level detection
- [x] Confluence scoring
- [x] Signal detection
- [x] Grid trading
- [x] Hedging
- [x] DCA/Martingale
- [x] Risk management
- [x] Main trading loop
- [x] Logging system
- [x] Professional GUI
- [x] CLI interface

### Future Enhancements 🚀
- [ ] Backtesting engine
- [ ] Strategy optimization
- [ ] ML-based signal filtering
- [ ] Multi-account support
- [ ] Telegram notifications
- [ ] Web dashboard
- [ ] Performance analytics
- [ ] Trade journal

## Support

For issues, questions, or contributions:
1. Check the logs in `logs/` directory
2. Review `trading_bot.log` for errors
3. Check `trades.log` for trade history
4. Check `signals.log` for signal history

## License

This project is for educational and research purposes.
Use at your own risk.

## Disclaimer

Trading forex and CFDs carries a high level of risk and may not be suitable for all investors.
Past performance is not indicative of future results.
This software is provided "as is" without warranty of any kind.
The authors are not responsible for any losses incurred through the use of this software.

---

**Built with data from EA analysis of 428 trades with 64.3% win rate**
