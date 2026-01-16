"""
Professional Trading Bot GUI - Redesigned
Clean, modern interface for monitoring and control
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from datetime import datetime
from typing import Optional

from core.mt5_manager import MT5Manager
from strategies.confluence_strategy import ConfluenceStrategy
from utils.credential_store import CredentialStore
from utils.timezone_manager import get_timezone_manager, format_trading_time
from portfolio.portfolio_manager import PortfolioManager


class TradingGUI:
    """Main GUI application for trading bot"""

    def __init__(self, root):
        """Initialize GUI"""
        self.root = root
        self.root.title("Confluence Trading Bot - Professional Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')

        # Set minimum window size
        self.root.minsize(1200, 800)

        # Bot components
        self.mt5_manager: Optional[MT5Manager] = None
        self.strategy: Optional[ConfluenceStrategy] = None
        self.strategy_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.credential_store = CredentialStore()

        # Create GUI
        self._create_gui()

        # Load saved credentials
        self._load_saved_credentials()

        # Start update loop
        self.root.after(1000, self._update_display)

    def _create_gui(self):
        """Create the complete GUI"""

        # ============================================================
        # HEADER SECTION
        # ============================================================
        header_frame = tk.Frame(self.root, bg='#1e1e1e')
        header_frame.pack(fill='x', padx=0, pady=0)

        title_label = tk.Label(
            header_frame,
            text=" CONFLUENCE TRADING BOT",
            font=('Arial', 22, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title_label.pack(pady=(15, 5))

        subtitle_label = tk.Label(
            header_frame,
            text="AI-Powered Multi-Timeframe Strategy | 64.3% Win Rate",
            font=('Arial', 11),
            bg='#1e1e1e',
            fg='#888888'
        )
        subtitle_label.pack(pady=(0, 15))

        # ============================================================
        # MAIN CONTENT AREA
        # ============================================================
        content_frame = tk.Frame(self.root, bg='#2b2b2b')
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)

        # LEFT COLUMN - Connection & Controls
        left_column = tk.Frame(content_frame, bg='#2b2b2b', width=320)
        left_column.pack(side='left', fill='y', padx=(0, 10))
        left_column.pack_propagate(False)

        # Connection Panel
        self._create_connection_panel(left_column)

        # Control Panel
        self._create_control_panel(left_column)

        # Account Info Panel
        self._create_account_panel(left_column)

        # RIGHT COLUMN - Main Display
        right_column = tk.Frame(content_frame, bg='#2b2b2b')
        right_column.pack(side='left', fill='both', expand=True, padx=(10, 0))

        # Stats Row
        stats_row = tk.Frame(right_column, bg='#2b2b2b')
        stats_row.pack(fill='x', pady=(0, 15))

        self._create_stats_panel(stats_row)
        self._create_risk_panel(stats_row)

        # Trading Windows Panel
        self._create_portfolio_panel(right_column)

        # Positions Table
        self._create_positions_panel(right_column)

        # Activity Log
        self._create_log_panel(right_column)

    def _create_connection_panel(self, parent):
        """Create MT5 connection panel"""
        panel = tk.LabelFrame(
            parent,
            text="  MT5 CONNECTION  ",
            font=('Arial', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ffffff',
            bd=0,
            relief='flat',
            padx=5,
            pady=5
        )
        panel.pack(fill='x', pady=(0, 15), padx=0, ipady=5)

        # Login
        tk.Label(panel, text="Login:", bg='#3a3a3a', fg='#cccccc', font=('Arial', 9)).pack(anchor='w', padx=15, pady=(10, 2))
        self.login_entry = tk.Entry(panel, bg='#2b2b2b', fg='#ffffff', insertbackground='#ffffff', relief='flat', font=('Arial', 10))
        self.login_entry.pack(fill='x', padx=15, pady=(0, 10))

        # Password
        tk.Label(panel, text="Password:", bg='#3a3a3a', fg='#cccccc', font=('Arial', 9)).pack(anchor='w', padx=15, pady=(0, 2))
        self.password_entry = tk.Entry(panel, show="●", bg='#2b2b2b', fg='#ffffff', insertbackground='#ffffff', relief='flat', font=('Arial', 10))
        self.password_entry.pack(fill='x', padx=15, pady=(0, 10))

        # Server
        tk.Label(panel, text="Server:", bg='#3a3a3a', fg='#cccccc', font=('Arial', 9)).pack(anchor='w', padx=15, pady=(0, 2))
        self.server_entry = tk.Entry(panel, bg='#2b2b2b', fg='#ffffff', insertbackground='#ffffff', relief='flat', font=('Arial', 10))
        self.server_entry.pack(fill='x', padx=15, pady=(0, 10))

        # Remember Me Checkbox
        self.remember_var = tk.BooleanVar(value=True)
        remember_check = tk.Checkbutton(
            panel,
            text="Remember credentials",
            variable=self.remember_var,
            bg='#3a3a3a',
            fg='#cccccc',
            selectcolor='#2b2b2b',
            activebackground='#3a3a3a',
            activeforeground='#ffffff',
            font=('Arial', 9)
        )
        remember_check.pack(anchor='w', padx=15, pady=(0, 10))

        # Buttons row
        button_frame = tk.Frame(panel, bg='#3a3a3a')
        button_frame.pack(fill='x', padx=15, pady=(0, 10))

        # Connect Button
        self.connect_btn = tk.Button(
            button_frame,
            text="CONNECT",
            command=self._connect_mt5,
            bg='#0078d4',
            fg='#ffffff',
            font=('Arial', 10, 'bold'),
            relief='flat',
            cursor='hand2',
            activebackground='#005a9e',
            width=12
        )
        self.connect_btn.pack(side='left', padx=(0, 5))

        # Clear Button
        clear_btn = tk.Button(
            button_frame,
            text="CLEAR",
            command=self._clear_credentials,
            bg='#5a5a5a',
            fg='#ffffff',
            font=('Arial', 10, 'bold'),
            relief='flat',
            cursor='hand2',
            activebackground='#4a4a4a',
            width=8
        )
        clear_btn.pack(side='left')

        # Status
        self.connection_status = tk.Label(
            panel,
            text="⚪ Not Connected",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 9)
        )
        self.connection_status.pack(pady=(0, 10))

    def _create_control_panel(self, parent):
        """Create bot control panel - simplified to show current settings"""
        panel = tk.LabelFrame(
            parent,
            text="  CURRENT SETTINGS  ",
            font=('Arial', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ffffff',
            bd=0,
            relief='flat',
            padx=5,
            pady=5
        )
        panel.pack(fill='x', pady=(0, 15), padx=0, ipady=5)

        # Current Lot Size (Read-only display)
        lot_frame = tk.Frame(panel, bg='#3a3a3a')
        lot_frame.pack(fill='x', padx=15, pady=(10, 8))

        tk.Label(
            lot_frame,
            text="Lot Size:",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 9)
        ).pack(side='left')

        from config.strategy_config import BASE_LOT_SIZE
        self.lot_size_label = tk.Label(
            lot_frame,
            text=f"{BASE_LOT_SIZE}",
            bg='#3a3a3a',
            fg='#00ff00',
            font=('Arial', 10, 'bold')
        )
        self.lot_size_label.pack(side='right')

        # Separator
        tk.Frame(panel, bg='#555555', height=1).pack(fill='x', padx=15, pady=5)

        # Trading Instruments (Dynamic - updates in real-time)
        instruments_frame = tk.Frame(panel, bg='#3a3a3a')
        instruments_frame.pack(fill='x', padx=15, pady=(8, 10))

        tk.Label(
            instruments_frame,
            text="Tradeable Now:",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 9)
        ).pack(anchor='w', pady=(0, 5))

        # Dynamic label - updates every second with currently tradeable instruments
        self.control_instruments_label = tk.Label(
            instruments_frame,
            text="Loading...",
            bg='#3a3a3a',
            fg='#00ff00',
            font=('Arial', 9, 'bold'),
            wraplength=250,
            justify='left'
        )
        self.control_instruments_label.pack(anchor='w')

        # Trend filter info
        tk.Label(
            panel,
            text=" Prevents trading in strong trending markets",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 8)
        ).pack(padx=15, pady=(0, 10))

        # Reload Config Button
        reload_config_btn = tk.Button(
            panel,
            text="🔄 RELOAD CONFIG",
            command=self._reload_config,
            bg='#0078d4',
            fg='#ffffff',
            font=('Arial', 9, 'bold'),
            relief='flat',
            cursor='hand2',
            activebackground='#106ebe'
        )
        reload_config_btn.pack(fill='x', padx=15, pady=(0, 5))

        # Reload info
        tk.Label(
            panel,
            text="↻ Reload settings without restarting bot",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 8)
        ).pack(padx=15, pady=(0, 10))

        # Start Button
        self.start_btn = tk.Button(
            panel,
            text="▶ START TRADING",
            command=self._start_bot,
            bg='#107c10',
            fg='#ffffff',
            font=('Arial', 10, 'bold'),
            relief='flat',
            cursor='hand2',
            state='disabled',
            activebackground='#0e6a0e'
        )
        self.start_btn.pack(fill='x', padx=15, pady=(0, 5))

        # Stop Button
        self.stop_btn = tk.Button(
            panel,
            text="⏹ STOP TRADING",
            command=self._stop_bot,
            bg='#e81123',
            fg='#ffffff',
            font=('Arial', 10, 'bold'),
            relief='flat',
            cursor='hand2',
            state='disabled',
            activebackground='#c50f1f'
        )
        self.stop_btn.pack(fill='x', padx=15, pady=(0, 10))

        # Bot Status
        self.bot_status = tk.Label(
            panel,
            text="⚪ Stopped",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 9)
        )
        self.bot_status.pack(pady=(0, 10))

    def _create_account_panel(self, parent):
        """Create account info panel"""
        panel = tk.LabelFrame(
            parent,
            text="  ACCOUNT INFO  ",
            font=('Arial', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ffffff',
            bd=0,
            relief='flat',
            padx=5,
            pady=5
        )
        panel.pack(fill='both', expand=True, pady=(0, 0), padx=0, ipady=5)

        self.account_labels = {}
        account_fields = [
            ('Balance', 'balance', '$'),
            ('Equity', 'equity', '$'),
            ('Margin', 'margin', '$'),
            ('Free Margin', 'free_margin', '$'),
            ('Profit/Loss', 'profit', '$')
        ]

        for label, key, prefix in account_fields:
            row_frame = tk.Frame(panel, bg='#3a3a3a')
            row_frame.pack(fill='x', padx=15, pady=3)

            tk.Label(
                row_frame,
                text=label + ":",
                bg='#3a3a3a',
                fg='#cccccc',
                font=('Arial', 9),
                anchor='w'
            ).pack(side='left')

            value_label = tk.Label(
                row_frame,
                text="--",
                bg='#3a3a3a',
                fg='#ffffff',
                font=('Arial', 9, 'bold'),
                anchor='e'
            )
            value_label.pack(side='right')
            self.account_labels[key] = (value_label, prefix)

    def _create_stats_panel(self, parent):
        """Create trading statistics panel"""
        panel = tk.LabelFrame(
            parent,
            text="  STATISTICS  ",
            font=('Arial', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ffffff',
            bd=0,
            relief='flat',
            padx=10,
            pady=5
        )
        panel.pack(side='left', fill='both', expand=True, padx=(0, 10))

        self.stats_labels = {}
        stats_fields = [
            ('Signals', 'signals_detected'),
            ('Trades Opened', 'trades_opened'),
            ('Trades Closed', 'trades_closed'),
            ('Grid Levels', 'grid_levels_added'),
            ('Hedges', 'hedges_activated'),
            ('DCA Levels', 'dca_levels_added')
        ]

        for label, key in stats_fields:
            row_frame = tk.Frame(panel, bg='#3a3a3a')
            row_frame.pack(fill='x', padx=15, pady=2)

            tk.Label(
                row_frame,
                text=label + ":",
                bg='#3a3a3a',
                fg='#cccccc',
                font=('Arial', 9)
            ).pack(side='left')

            value_label = tk.Label(
                row_frame,
                text="0",
                bg='#3a3a3a',
                fg='#00ff00',
                font=('Arial', 9, 'bold')
            )
            value_label.pack(side='right')
            self.stats_labels[key] = value_label

    def _create_risk_panel(self, parent):
        """Create risk metrics panel"""
        panel = tk.LabelFrame(
            parent,
            text="  RISK METRICS  ",
            font=('Arial', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ffffff',
            bd=0,
            relief='flat',
            padx=10,
            pady=5
        )
        panel.pack(side='left', fill='both', expand=True, padx=(10, 0))

        self.risk_labels = {}
        risk_fields = [
            ('Drawdown', 'drawdown_pct', '%'),
            ('Total Volume', 'total_volume', ' lots'),
            ('Exposure', 'exposure_pct', '%'),
            ('Open Positions', 'positions_count', '')
        ]

        for label, key, suffix in risk_fields:
            row_frame = tk.Frame(panel, bg='#3a3a3a')
            row_frame.pack(fill='x', padx=15, pady=2)

            tk.Label(
                row_frame,
                text=label + ":",
                bg='#3a3a3a',
                fg='#cccccc',
                font=('Arial', 9)
            ).pack(side='left')

            value_label = tk.Label(
                row_frame,
                text="--",
                bg='#3a3a3a',
                fg='#ffaa00',
                font=('Arial', 9, 'bold')
            )
            value_label.pack(side='right')
            self.risk_labels[key] = (value_label, suffix)

    def _create_portfolio_panel(self, parent):
        """Create portfolio/trading windows panel"""
        panel = tk.LabelFrame(
            parent,
            text="  TRADING WINDOWS (GMT/GMT+1)  ",
            font=('Arial', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ffffff',
            bd=0,
            relief='flat',
            padx=10,
            pady=10
        )
        panel.pack(fill='x', pady=(0, 15))

        # Create portfolio status display
        status_frame = tk.Frame(panel, bg='#3a3a3a')
        status_frame.pack(fill='x', padx=5, pady=5)

        # Current time display (timezone-aware)
        time_frame = tk.Frame(status_frame, bg='#3a3a3a')
        time_frame.pack(fill='x', pady=(0, 10))

        tk.Label(
            time_frame,
            text="Current Time:",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 9)
        ).pack(side='left')

        self.portfolio_time_label = tk.Label(
            time_frame,
            text="--:--:-- GMT",
            bg='#3a3a3a',
            fg='#00ff00',
            font=('Arial', 9, 'bold')
        )
        self.portfolio_time_label.pack(side='left', padx=(10, 0))

        # Tradeable instruments display
        instruments_frame = tk.Frame(status_frame, bg='#3a3a3a')
        instruments_frame.pack(fill='x', pady=(0, 10))

        tk.Label(
            instruments_frame,
            text="Tradeable Now:",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 9)
        ).pack(side='left')

        self.tradeable_instruments_label = tk.Label(
            instruments_frame,
            text="Loading...",
            bg='#3a3a3a',
            fg='#00ff00',
            font=('Arial', 9, 'bold')
        )
        self.tradeable_instruments_label.pack(side='left', padx=(10, 0))

        # Trading restrictions display
        restrictions_frame = tk.Frame(status_frame, bg='#3a3a3a')
        restrictions_frame.pack(fill='x')

        tk.Label(
            restrictions_frame,
            text="Status:",
            bg='#3a3a3a',
            fg='#888888',
            font=('Arial', 9)
        ).pack(side='left')

        self.trading_status_label = tk.Label(
            restrictions_frame,
            text="Checking...",
            bg='#3a3a3a',
            fg='#00ff00',
            font=('Arial', 9, 'bold')
        )
        self.trading_status_label.pack(side='left', padx=(10, 0))

        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager()

    def _create_positions_panel(self, parent):
        """Create positions table panel"""
        panel = tk.LabelFrame(
            parent,
            text="  OPEN POSITIONS  ",
            font=('Arial', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ffffff',
            bd=0,
            relief='flat',
            padx=5,
            pady=5
        )
        panel.pack(fill='both', expand=True, pady=(0, 15))

        # Style for treeview
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Custom.Treeview',
            background='#2b2b2b',
            foreground='#ffffff',
            fieldbackground='#2b2b2b',
            borderwidth=0,
            font=('Arial', 9)
        )
        style.configure('Custom.Treeview.Heading', background='#1e1e1e', foreground='#ffffff', font=('Arial', 9, 'bold'))
        style.map('Custom.Treeview', background=[('selected', '#0078d4')])

        # Create treeview
        columns = ('Ticket', 'Symbol', 'Type', 'Volume', 'Entry', 'Current', 'P/L', 'Recovery')
        self.positions_tree = ttk.Treeview(
            panel,
            columns=columns,
            show='headings',
            height=6,
            style='Custom.Treeview'
        )

        # Configure columns
        widths = {'Ticket': 80, 'Symbol': 80, 'Type': 60, 'Volume': 70, 'Entry': 90, 'Current': 90, 'P/L': 80, 'Recovery': 100}
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=widths[col], anchor='center')

        # Scrollbar
        scrollbar = ttk.Scrollbar(panel, orient='vertical', command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=scrollbar.set)

        self.positions_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y', padx=(0, 10), pady=10)

    def _create_log_panel(self, parent):
        """Create activity log panel"""
        panel = tk.LabelFrame(
            parent,
            text="  ACTIVITY LOG  ",
            font=('Arial', 10, 'bold'),
            bg='#3a3a3a',
            fg='#ffffff',
            bd=0,
            relief='flat',
            padx=5,
            pady=5
        )
        panel.pack(fill='both', expand=True)

        self.log_text = tk.Text(
            panel,
            height=8,
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Consolas', 9),
            relief='flat',
            wrap='word'
        )
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = tk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

    def _load_saved_credentials(self):
        """Load saved credentials on startup"""
        credentials = self.credential_store.load_credentials()
        if credentials:
            self.login_entry.insert(0, credentials['login'])
            self.password_entry.insert(0, credentials['password'])
            self.server_entry.insert(0, credentials['server'])
            self._log(" Loaded saved credentials")

    def _clear_credentials(self):
        """Clear saved credentials and form"""
        self.credential_store.clear_credentials()
        self.login_entry.delete(0, 'end')
        self.password_entry.delete(0, 'end')
        self.server_entry.delete(0, 'end')
        self._log("🗑️ Credentials cleared")

    def _connect_mt5(self):
        """Connect to MT5"""
        login = self.login_entry.get()
        password = self.password_entry.get()
        server = self.server_entry.get()

        if not all([login, password, server]):
            messagebox.showerror("Error", "Please fill in all connection fields")
            return

        try:
            login_int = int(login)
        except ValueError:
            messagebox.showerror("Error", "Login must be a number")
            return

        self._log("Connecting to MT5...")

        self.mt5_manager = MT5Manager(login_int, password, server)

        if self.mt5_manager.connect():
            self.connection_status.config(text="🟢 Connected", fg='#00ff00')
            self.start_btn.config(state='normal')
            self._log("[OK] Connected to MT5 successfully")
            self.strategy = ConfluenceStrategy(self.mt5_manager)

            # Save credentials if remember is checked
            if self.remember_var.get():
                self.credential_store.save_credentials(
                    login=login,
                    password=password,
                    server=server,
                    remember=True
                )
                self._log(" Credentials saved")
        else:
            self.connection_status.config(text="🔴 Failed", fg='#ff0000')
            self._log("[ERROR] Failed to connect to MT5")

    def _reload_config(self):
        """Reload configuration without restarting bot"""
        try:
            # Use the config reloader utility
            from utils.config_reloader import reload_config, print_current_config, clear_pycache

            self._log("🔄 Reloading configuration...")

            # Clear Python bytecode cache
            clear_pycache()

            # Reload the config module
            success = reload_config()

            if success:
                self._log("[OK] Configuration reloaded successfully!")

                # If strategy is running, call its reload method
                if self.strategy:
                    self.strategy.reload_config()
                    self._log("[OK] Strategy configuration updated")
                else:
                    self._log("ℹ️ Strategy not running - changes will apply on next start")

                # Update GUI fields with new values
                from config.strategy_config import BASE_LOT_SIZE, MAX_DRAWDOWN_PERCENT, TREND_FILTER_ENABLED

                self.lot_size_entry.delete(0, tk.END)
                self.lot_size_entry.insert(0, str(BASE_LOT_SIZE))

                self.drawdown_entry.delete(0, tk.END)
                self.drawdown_entry.insert(0, str(MAX_DRAWDOWN_PERCENT))

                self.trend_filter_var.set(TREND_FILTER_ENABLED)

                messagebox.showinfo(
                    "Config Reloaded",
                    "[OK] Configuration reloaded successfully!\n\n"
                    "Changes will take effect:\n"
                    "* Immediately for new trades\n"
                    "* On next trading cycle\n\n"
                    "No restart needed!"
                )
            else:
                self._log("[ERROR] Failed to reload configuration")
                messagebox.showerror("Error", "Failed to reload configuration")

        except Exception as e:
            self._log(f"[ERROR] Config reload error: {str(e)}")
            messagebox.showerror("Error", f"Failed to reload config: {str(e)}")

    def _start_bot(self):
        """Start the trading bot"""
        if not self.strategy:
            messagebox.showerror("Error", "Please connect to MT5 first")
            return

        # Get symbols from portfolio configuration
        from portfolio.instruments_config import get_enabled_instruments
        symbols = get_enabled_instruments()

        if not symbols:
            messagebox.showerror("Error", "No instruments configured in portfolio")
            return

        self._log(f" Starting bot with instruments: {', '.join(symbols)}")
        self._log(f"   Trading windows will be enforced for each instrument")

        self.is_running = True
        self.strategy_thread = threading.Thread(
            target=self.strategy.start,
            args=(symbols,),
            daemon=True
        )
        self.strategy_thread.start()

        self.bot_status.config(text="🟢 Running", fg='#00ff00')
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

    def _stop_bot(self):
        """Stop the trading bot"""
        if self.strategy:
            self._log("⏹ Stopping bot...")
            self.strategy.stop()

        self.is_running = False
        self.bot_status.config(text="🔴 Stopped", fg='#ff0000')
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def _update_display(self):
        """Update all display elements"""
        if self.strategy and self.is_running:
            try:
                status = self.strategy.get_status()

                # Update account info
                if status['account']:
                    acc = status['account']
                    self.account_labels['balance'][0].config(text=f"${acc.get('balance', 0):.2f}")
                    self.account_labels['equity'][0].config(text=f"${acc.get('equity', 0):.2f}")
                    self.account_labels['margin'][0].config(text=f"${acc.get('margin', 0):.2f}")
                    self.account_labels['free_margin'][0].config(text=f"${acc.get('free_margin', 0):.2f}")

                    profit = acc.get('profit', 0)
                    profit_color = '#00ff00' if profit >= 0 else '#ff0000'
                    self.account_labels['profit'][0].config(text=f"${profit:.2f}", fg=profit_color)

                # Update statistics
                stats = status['statistics']
                for key, label in self.stats_labels.items():
                    label.config(text=str(stats.get(key, 0)))

                # Update risk metrics
                risk = status['risk_metrics']
                dd = risk.get('drawdown_pct', 0)
                dd_color = '#ff0000' if dd > 5 else '#ffaa00' if dd > 2 else '#00ff00'
                self.risk_labels['drawdown_pct'][0].config(text=f"{dd:.1f}%", fg=dd_color)
                self.risk_labels['total_volume'][0].config(text=f"{risk.get('total_volume', 0):.2f}")
                self.risk_labels['exposure_pct'][0].config(text=f"{risk.get('exposure_pct', 0):.1f}%")
                self.risk_labels['positions_count'][0].config(text=str(risk.get('positions_count', 0)))

                # Update portfolio/trading windows panel
                self._update_portfolio_panel()

                # Update positions
                self._update_positions_table(status['positions'], status['recovery_status'])

            except Exception as e:
                self._log(f"[ERROR] Error updating display: {e}")

        self.root.after(1000, self._update_display)

    def _update_portfolio_panel(self):
        """Update portfolio/trading windows panel with current status"""
        try:
            tz_mgr = get_timezone_manager()
            current_time = tz_mgr.get_current_trading_timezone()

            # Update time display with timezone
            time_str = format_trading_time(current_time)
            self.portfolio_time_label.config(text=time_str)

            # Get tradeable instruments
            if hasattr(self, 'portfolio_manager'):
                tradeable_symbols = self.portfolio_manager.get_tradeable_symbols(current_time)

                if tradeable_symbols:
                    instruments_text = ", ".join(tradeable_symbols)

                    # Update portfolio panel
                    self.tradeable_instruments_label.config(
                        text=instruments_text,
                        fg='#00ff00'
                    )

                    # Update control panel instruments (dynamic)
                    if hasattr(self, 'control_instruments_label'):
                        self.control_instruments_label.config(
                            text=instruments_text,
                            fg='#00ff00'
                        )
                else:
                    # No instruments tradeable right now
                    self.tradeable_instruments_label.config(
                        text="None (outside trading windows)",
                        fg='#ffaa00'
                    )

                    # Update control panel
                    if hasattr(self, 'control_instruments_label'):
                        self.control_instruments_label.config(
                            text="None (outside windows)",
                            fg='#ffaa00'
                        )

                # Check trading calendar status
                from utils.trading_calendar import get_trading_calendar
                calendar = get_trading_calendar()
                is_allowed, reason = calendar.is_trading_allowed(current_time)

                if is_allowed:
                    self.trading_status_label.config(
                        text="[OK] Trading Allowed",
                        fg='#00ff00'
                    )
                else:
                    self.trading_status_label.config(
                        text=f"[ERROR] {reason}",
                        fg='#ff0000'
                    )
            else:
                self.tradeable_instruments_label.config(text="Portfolio manager not initialized")
                self.trading_status_label.config(text="N/A")

        except Exception as e:
            self.portfolio_time_label.config(text="Error")
            self.tradeable_instruments_label.config(text=f"Error: {str(e)}")
            self.trading_status_label.config(text="Error")

    def _update_positions_table(self, positions, recovery_status):
        """Update positions treeview"""
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)

        for pos in positions:
            ticket = pos['ticket']
            recovery_info = next((r for r in recovery_status if r['ticket'] == ticket), None)

            recovery_text = ""
            if recovery_info and recovery_info['recovery_active']:
                parts = []
                if recovery_info['grid_levels'] > 0:
                    parts.append(f"G:{recovery_info['grid_levels']}")
                if recovery_info['hedges_active'] > 0:
                    parts.append("H")
                if recovery_info['dca_levels'] > 0:
                    parts.append(f"D:{recovery_info['dca_levels']}")
                recovery_text = " ".join(parts)

            self.positions_tree.insert('', 'end', values=(
                ticket,
                pos['symbol'],
                pos['type'].upper(),
                f"{pos['volume']:.2f}",
                f"{pos['price_open']:.5f}",
                f"{pos['price_current']:.5f}",
                f"${pos['profit']:.2f}",
                recovery_text
            ))

    def _log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert('end', log_message)
        self.log_text.see('end')

    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Bot is running. Stop and quit?"):
                self._stop_bot()
                if self.mt5_manager:
                    self.mt5_manager.disconnect()
                self.root.destroy()
        else:
            if self.mt5_manager:
                self.mt5_manager.disconnect()
            self.root.destroy()


def main():
    """Run the GUI"""
    root = tk.Tk()
    app = TradingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
