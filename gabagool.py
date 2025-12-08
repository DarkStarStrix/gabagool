#!/usr/bin/env python3
"""
GabagoolBot Ultra - Polymarket Arbitrage Bot
Uses py_clob_client for reliable order execution
"""

import os
import json
import asyncio
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from dotenv import load_dotenv

# Load .env FIRST
load_dotenv(override=True)

from eth_account import Account
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, ApiCreds
from py_clob_client.order_builder.constants import BUY

# ========================== ASYNC LOGGING ==========================

import queue
import threading

class AsyncLogger:
    """Non-blocking logger - writes to queue, background thread handles I/O"""
    
    def __init__(self, name: str = 'gabagool', log_file: str = 'gabagool_ultra.log'):
        self.queue = queue.Queue()
        self.running = True
        
        # Setup formatters
        self.formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler only (console is slow)
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setFormatter(self.formatter)
        
        # Optional console (can disable for speed)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatter)
        self.console_enabled = True
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _writer_loop(self):
        """Background thread that handles actual I/O"""
        while self.running or not self.queue.empty():
            try:
                record = self.queue.get(timeout=0.1)
                if record is None:
                    break
                
                # Write to file (always)
                self.file_handler.emit(record)
                
                # Write to console (optional)
                if self.console_enabled:
                    self.console_handler.emit(record)
                    
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def _log(self, level: int, msg: str):
        """Non-blocking log - just puts in queue"""
        record = logging.LogRecord(
            name='gabagool',
            level=level,
            pathname='',
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None
        )
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass  # Drop log rather than block
    
    def info(self, msg: str):
        self._log(logging.INFO, msg)
    
    def warning(self, msg: str):
        self._log(logging.WARNING, msg)
    
    def error(self, msg: str):
        self._log(logging.ERROR, msg)
    
    def debug(self, msg: str):
        self._log(logging.DEBUG, msg)
    
    def stop(self):
        """Stop background writer"""
        self.running = False
        self.queue.put(None)
        self.writer_thread.join(timeout=2)

# Global async logger
_console_enabled = os.getenv('LOG_TO_CONSOLE', 'true').lower() == 'true'
logger = AsyncLogger()
logger.console_enabled = _console_enabled

# ========================== CONFIG ==========================

@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'SOL', 'XRP'])
    
    max_combined_price: float = 0.97
    yes_cheap_threshold: float = 0.48
    no_cheap_threshold: float = 0.48
    
    max_position_usd: float = 100.0
    min_order_usd: float = 5.0
    max_order_usd: float = 25.0
    max_imbalance_ratio: float = 1.3
    
    gamma_url: str = "https://gamma-api.polymarket.com"
    wss_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    
    wss_reconnect_delay: int = 2
    market_refresh_interval: int = 15  # Check for new 15-min slot every 15 seconds
    
    simulation_mode: bool = True
    
    @classmethod
    def from_env(cls) -> 'Config':
        markets = os.getenv('MARKETS', 'BTC,ETH,SOL,XRP')
        
        # Parse simulation mode - be explicit
        sim_raw = os.getenv('SIMULATION_MODE', 'true')
        # Remove any comments or whitespace
        sim_clean = sim_raw.split('#')[0].strip().lower()
        simulation = sim_clean in ('true', '1', 'yes', '')
        
        return cls(
            symbols=[m.strip().upper() for m in markets.split(',') if m.strip()],
            max_combined_price=float(os.getenv('MAX_COMBINED_PRICE', '0.97')),
            yes_cheap_threshold=float(os.getenv('YES_CHEAP_THRESHOLD', '0.48')),
            no_cheap_threshold=float(os.getenv('NO_CHEAP_THRESHOLD', '0.48')),
            max_position_usd=float(os.getenv('MAX_POSITION_USD', '100')),
            min_order_usd=float(os.getenv('MIN_ORDER_USD', '5')),
            max_order_usd=float(os.getenv('MAX_ORDER_USD', '25')),
            simulation_mode=simulation,
        )

# ========================== POSITION ==========================

@dataclass 
class Position:
    market_id: str
    yes_token_id: str
    no_token_id: str
    symbol: str = ""
    
    yes_shares: float = 0.0
    no_shares: float = 0.0
    yes_cost: float = 0.0
    no_cost: float = 0.0
    trades_count: int = 0
    
    @property
    def total_cost(self) -> float:
        return self.yes_cost + self.no_cost
    
    @property
    def avg_yes_price(self) -> float:
        return self.yes_cost / self.yes_shares if self.yes_shares > 0 else 0
    
    @property
    def avg_no_price(self) -> float:
        return self.no_cost / self.no_shares if self.no_shares > 0 else 0
    
    @property
    def avg_combined_price(self) -> float:
        total_shares = min(self.yes_shares, self.no_shares)
        if total_shares == 0:
            return 0
        return self.total_cost / total_shares
    
    @property
    def guaranteed_profit(self) -> float:
        total_shares = min(self.yes_shares, self.no_shares)
        if self.avg_combined_price >= 1 or total_shares == 0:
            return 0
        return total_shares * (1.0 - self.avg_combined_price)

# ========================== ORDERBOOK ==========================

class OrderbookCache:
    def __init__(self):
        self.books: Dict[str, dict] = {}
        self.timestamps: Dict[str, float] = {}
    
    def update(self, token_id: str, bids: list, asks: list):
        self.books[token_id] = {
            'bids': sorted(
                [{'price': float(b.get('price', 0)), 'size': float(b.get('size', 0))} for b in bids if b],
                key=lambda x: x['price'], reverse=True
            ),
            'asks': sorted(
                [{'price': float(a.get('price', 0)), 'size': float(a.get('size', 0))} for a in asks if a],
                key=lambda x: x['price']
            )
        }
        self.timestamps[token_id] = time.time()
    
    def get(self, token_id: str) -> dict:
        return self.books.get(token_id, {'bids': [], 'asks': []})

# ========================== MAIN BOT ==========================

class GabagoolUltra:
    def __init__(self, config: Config):
        self.config = config
        self.start_time = time.time()
        
        # Credentials
        private_key = os.getenv('POLYMARKET_PRIVATE_KEY', '')
        if not private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY required")
        if not private_key.startswith('0x'):
            private_key = '0x' + private_key
            
        funder = os.getenv('POLYMARKET_FUNDER_ADDRESS', '')
        
        self.account = Account.from_key(private_key)
        self.wallet = self.account.address
        
        if not funder:
            funder = self.wallet
        
        # CLOB Client (py_clob_client - proven to work)
        self.clob = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            signature_type=2,
            funder=funder
        )
        self._setup_api_creds()
        
        # Thread pool for sync CLOB calls
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # State
        self.orderbook = OrderbookCache()
        self.active_markets: Dict[str, dict] = {}
        self.positions: Dict[str, Position] = {}
        self.token_to_market: Dict[str, str] = {}
        
        self.running = False
        self.ws = None
        self.force_reconnect = False
        
        # Stats
        self.stats = {
            'wss_messages': 0,
            'wss_reconnects': 0,
            'opportunities': 0,
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_failed': 0,
            'total_spent': 0.0,
            'latencies': [],
            'clob_latencies': [],
        }
        
        # Database
        self._init_db()
        
        # Log startup
        mode = "🔬 SIMULATION" if config.simulation_mode else "🚀 PRODUCTION"
        logger.info("=" * 70)
        logger.info(f"GABAGOOL ULTRA [{mode}]")
        logger.info("=" * 70)
        logger.info(f"  Wallet: {self.wallet}")
        logger.info(f"  Symbols: {', '.join(config.symbols)}")
        logger.info(f"  Strategy: buy when < {config.yes_cheap_threshold}, combined < {config.max_combined_price}")
        logger.info(f"  Limits: ${config.min_order_usd}-${config.max_order_usd}, max ${config.max_position_usd}")
        if config.simulation_mode:
            logger.info("  ⚠️  SIMULATION MODE - no real orders will be placed")
        else:
            logger.info("  ⚠️  PRODUCTION MODE - real money at risk!")
        logger.info("=" * 70)
    
    def _setup_api_creds(self):
        """Load or generate API credentials"""
        api_key = os.getenv('API_KEY', '')
        api_secret = os.getenv('API_SECRET', '')
        api_passphrase = os.getenv('API_PASSPHRASE', '')
        
        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)
            self.clob.set_api_creds(creds)
            logger.info("✓ API credentials loaded from .env")
        else:
            try:
                creds = self.clob.create_or_derive_api_creds()
                self.clob.set_api_creds(creds)
                logger.info("✓ API credentials derived")
                logger.info(f"  Save these to .env:")
                logger.info(f"  API_KEY={creds.api_key}")
                logger.info(f"  API_SECRET={creds.api_secret}")
                logger.info(f"  API_PASSPHRASE={creds.api_passphrase}")
            except Exception as e:
                logger.error(f"API credentials error: {e}")
    
    def _init_db(self):
        self.db = sqlite3.connect('gabagool_ultra.db')
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY, timestamp TEXT, symbol TEXT,
                side TEXT, price REAL, shares REAL, cost REAL,
                latency_ms REAL, clob_latency_ms REAL,
                combined_price REAL, profit REAL, order_id TEXT
            )
        ''')
        self.db.commit()
    
    # ==================== MARKET DISCOVERY ====================
    
    def get_slug(self, symbol: str) -> str:
        from zoneinfo import ZoneInfo
        now = datetime.now(tz=ZoneInfo("America/New_York"))
        slot = (now.minute // 15) * 15
        ts = int(now.replace(minute=slot, second=0, microsecond=0)
                 .astimezone(ZoneInfo("UTC")).timestamp())
        return f"{symbol.lower()}-updown-15m-{ts}"
    
    async def fetch_markets(self):
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_market(session, s) for s in self.config.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            new_markets = {}
            for result in results:
                if isinstance(result, dict) and 'slug' in result:
                    market_id = result['slug']
                    new_markets[market_id] = result
            
            # Check if markets changed
            old_market_ids = set(self.active_markets.keys())
            new_market_ids = set(new_markets.keys())
            
            markets_changed = old_market_ids != new_market_ids
            
            if markets_changed:
                logger.info(f"🔄 Markets changed: {list(old_market_ids)} -> {list(new_market_ids)}")
                
                # Only rebuild when changed
                self.active_markets = new_markets
                self.token_to_market.clear()
                self.positions.clear()
                
                for market_id, market in new_markets.items():
                    yes_t = market['yes_token_id']
                    no_t = market['no_token_id']
                    
                    self.token_to_market[yes_t] = market_id
                    self.token_to_market[no_t] = market_id
                    
                    self.positions[market_id] = Position(
                        market_id=market_id,
                        yes_token_id=yes_t,
                        no_token_id=no_t,
                        symbol=market['symbol']
                    )
            
            return markets_changed
    
    async def _fetch_market(self, session, symbol: str) -> Optional[dict]:
        slug = self.get_slug(symbol)
        url = f"{self.config.gamma_url}/markets/slug/{slug}"
        
        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    clob_ids = data.get("clobTokenIds") or data.get("clob_token_ids")
                    
                    if isinstance(clob_ids, str):
                        try:
                            clob_ids = json.loads(clob_ids)
                        except:
                            clob_ids = [x.strip().strip('"') for x in clob_ids.strip("[]").split(",")]
                    
                    if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                        data['symbol'] = symbol
                        data['slug'] = slug
                        data['yes_token_id'] = clob_ids[0]
                        data['no_token_id'] = clob_ids[1]
                        logger.info(f"[{symbol}] {slug}")
                        return data
        except Exception as e:
            logger.debug(f"[{symbol}] fetch error: {e}")
        return None
    
    # ==================== WEBSOCKET ====================
    
    async def subscribe(self, token_ids: List[str]):
        if not self.ws or not token_ids:
            return
        try:
            await self.ws.send_json({"assets_ids": token_ids, "type": "market"})
            logger.info(f"Subscribed to {len(token_ids)} tokens")
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
    
    async def handle_message(self, data: dict, recv_time: float):
        self.stats['wss_messages'] += 1
        
        event = data.get('event_type')
        
        if event == 'book':
            token_id = data.get('asset_id')
            bids = data.get('bids') or data.get('buys', [])
            asks = data.get('asks') or data.get('sells', [])
            
            if token_id:
                self.orderbook.update(token_id, bids, asks)
                market_id = self.token_to_market.get(token_id)
                if market_id:
                    await self.analyze(market_id, recv_time)
        
        elif event == 'price_change':
            for change in data.get('price_changes', []):
                token_id = change.get('asset_id')
                if token_id:
                    market_id = self.token_to_market.get(token_id)
                    if market_id:
                        await self.analyze(market_id, recv_time)
    
    async def wss_loop(self):
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        self.config.wss_url,
                        heartbeat=30,
                        receive_timeout=60
                    ) as ws:
                        self.ws = ws
                        logger.info("WSS connected")
                        
                        # Subscribe to all tokens
                        tokens = []
                        for m in self.active_markets.values():
                            tokens.extend([m.get('yes_token_id'), m.get('no_token_id')])
                        tokens = [t for t in tokens if t]
                        if tokens:
                            await self.subscribe(tokens)
                        
                        async for msg in ws:
                            if not self.running:
                                break
                            
                            # Check for forced reconnect (new market slot)
                            if self.force_reconnect:
                                logger.info("Force reconnect requested")
                                self.force_reconnect = False
                                break
                            
                            recv_time = time.time()
                            
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    if isinstance(data, list):
                                        for event in data:
                                            if isinstance(event, dict):
                                                await self.handle_message(event, recv_time)
                                    elif isinstance(data, dict):
                                        await self.handle_message(data, recv_time)
                                except:
                                    pass
                            
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                                break
            
            except Exception as e:
                logger.error(f"WSS error: {e}")
            
            self.ws = None
            if self.running:
                self.stats['wss_reconnects'] += 1
                logger.info(f"WSS reconnecting... (reconnect #{self.stats['wss_reconnects']})")
                await asyncio.sleep(self.config.wss_reconnect_delay)
    
    # ==================== STRATEGY ====================
    
    async def analyze(self, market_id: str, recv_time: float):
        pos = self.positions.get(market_id)
        if not pos:
            return
        
        yes_book = self.orderbook.get(pos.yes_token_id)
        no_book = self.orderbook.get(pos.no_token_id)
        
        yes_asks = yes_book.get('asks', [])
        no_asks = no_book.get('asks', [])
        
        if not yes_asks or not no_asks:
            return
        
        best_yes = round(yes_asks[0]['price'], 2)
        best_no = round(no_asks[0]['price'], 2)
        combined = best_yes + best_no
        
        # Log prices rarely (every 1000 messages)
        if self.stats['wss_messages'] % 1000 == 0:
            arb_status = "✓ ARB" if combined < self.config.max_combined_price else "✗ NO ARB"
            logger.info(f"[{pos.symbol}] UP={best_yes:.3f} DOWN={best_no:.3f} Combined={combined:.3f} [{arb_status}]")
        
        # CRITICAL: Only trade when combined < max_combined_price (arbitrage condition)
        # Add buffer for price slippage (we'll pay slightly more)
        price_buffer = 0.02  # 2 cent buffer per side to ensure fills
        combined_with_buffer = combined + (price_buffer * 2)
        
        if combined_with_buffer >= self.config.max_combined_price:
            return  # No arbitrage after accounting for slippage
        
        # Position limits
        if pos.total_cost >= self.config.max_position_usd:
            return
        
        # Calculate how much to buy - need EQUAL shares on both sides
        remaining = self.config.max_position_usd - pos.total_cost
        
        # Cost per pair = best_yes + best_no (combined price)
        cost_per_pair = combined
        max_pairs = remaining / cost_per_pair
        
        # Limit by max_order (considering both sides)
        max_pairs_by_order = (self.config.max_order_usd * 2) / cost_per_pair
        shares = min(max_pairs, max_pairs_by_order)
        
        # Round DOWN to whole number (Polymarket requires clean amounts)
        shares = int(shares)
        
        # Must have at least 1 share
        if shares < 1:
            return
        
        # Check minimum PER SIDE (not total) - use buffered prices
        buy_yes_price = round(best_yes + price_buffer, 2)
        buy_no_price = round(best_no + price_buffer, 2)
        
        if shares * buy_yes_price < self.config.min_order_usd or shares * buy_no_price < self.config.min_order_usd:
            logger.info(f"[{pos.symbol}] Order too small: YES=${shares * buy_yes_price:.2f} NO=${shares * buy_no_price:.2f} (min=${self.config.min_order_usd})")
            return
        
        # ARBITRAGE FOUND! All checks passed
        self.stats['opportunities'] += 1
        profit_margin = (1.0 - combined) * 100
        logger.info(f"[{pos.symbol}] 💰 ARBITRAGE: Combined={combined:.4f} Margin={profit_margin:.2f}%")
        
        # Buy BOTH sides with EQUAL shares (prices already buffered above)
        latency = (time.time() - recv_time) * 1000
        yes_cost = shares * buy_yes_price
        no_cost = shares * buy_no_price
        total_cost = yes_cost + no_cost
        expected_profit = shares * (1.0 - buy_yes_price - buy_no_price)
        
        logger.info(f"[{pos.symbol}] ⚡ BUY PAIR: {shares} shares @ {buy_yes_price + buy_no_price:.4f} = ${total_cost:.2f} | Expected profit: ${expected_profit:.2f} | {latency:.1f}ms")
        
        # Execute pair with FOK and partial fill handling
        await self.execute_pair(pos, buy_yes_price, buy_no_price, shares, recv_time)
    
    def _can_buy(self, pos: Position, side: str) -> bool:
        if side == 'YES':
            if pos.yes_shares == 0:
                return True
            if pos.no_shares == 0:
                return pos.yes_shares < self.config.max_position_usd / 2
            return pos.yes_shares <= pos.no_shares * self.config.max_imbalance_ratio
        else:
            if pos.no_shares == 0:
                return True
            if pos.yes_shares == 0:
                return pos.no_shares < self.config.max_position_usd / 2
            return pos.no_shares <= pos.yes_shares * self.config.max_imbalance_ratio
    
    def _improves(self, pos: Position, side: str, price: float) -> bool:
        test = self.config.min_order_usd / price
        
        if side == 'YES':
            new_avg = (pos.yes_cost + test * price) / (pos.yes_shares + test)
            if pos.no_shares > 0:
                return new_avg + pos.avg_no_price <= self.config.max_combined_price
            return new_avg <= self.config.max_combined_price / 2
        else:
            new_avg = (pos.no_cost + test * price) / (pos.no_shares + test)
            if pos.yes_shares > 0:
                return pos.avg_yes_price + new_avg <= self.config.max_combined_price
            return new_avg <= self.config.max_combined_price / 2
    
    # ==================== EXECUTION ====================
    
    async def execute_pair(self, pos: Position, yes_price: float, no_price: float, shares: float, recv_time: float):
        """Execute arbitrage pair - buy both sides with FOK"""
        
        yes_token = pos.yes_token_id
        no_token = pos.no_token_id
        
        if self.config.simulation_mode:
            # Simulate both orders
            await self._execute_sim(pos, 'YES', yes_price, shares, shares * yes_price, recv_time)
            await self._execute_sim(pos, 'NO', no_price, shares, shares * no_price, recv_time)
            return
        
        # Real execution - FOK orders in parallel
        results = await asyncio.gather(
            self._execute_fok(pos, 'YES', yes_price, shares, yes_token, recv_time),
            self._execute_fok(pos, 'NO', no_price, shares, no_token, recv_time),
            return_exceptions=True
        )
        
        yes_result, no_result = results
        
        yes_filled = isinstance(yes_result, dict) and yes_result.get('filled', False)
        no_filled = isinstance(no_result, dict) and no_result.get('filled', False)
        
        if yes_filled and no_filled:
            # Perfect - both sides filled
            logger.info(f"[{pos.symbol}] ✓✓ PAIR COMPLETE: UP + DOWN filled")
            combined = pos.avg_combined_price
            profit = pos.guaranteed_profit
            logger.info(f"[{pos.symbol}] Combined: {combined:.4f} | Guaranteed profit: ${profit:.2f}")
            
        elif yes_filled and not no_filled:
            # Problem - only YES filled
            logger.warning(f"[{pos.symbol}] ⚠️ PARTIAL: Only UP filled, DOWN failed!")
            logger.warning(f"[{pos.symbol}] Attempting to sell UP to close position...")
            await self._emergency_sell(pos, 'YES', yes_price, shares)
            
        elif no_filled and not yes_filled:
            # Problem - only NO filled
            logger.warning(f"[{pos.symbol}] ⚠️ PARTIAL: Only DOWN filled, UP failed!")
            logger.warning(f"[{pos.symbol}] Attempting to sell DOWN to close position...")
            await self._emergency_sell(pos, 'NO', no_price, shares)
            
        else:
            # Both failed - no position, no problem
            logger.info(f"[{pos.symbol}] Both orders failed/cancelled - no position taken")
    
    async def _execute_fok(self, pos: Position, side: str, price: float, shares: float, token_id: str, recv_time: float) -> dict:
        """Execute single FOK order"""
        self.stats['orders_sent'] += 1
        clob_start = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._send_fok_order_sync,
                token_id, price, shares, side
            )
            
            clob_latency = (time.time() - clob_start) * 1000
            total_latency = (time.time() - recv_time) * 1000
            
            self.stats['clob_latencies'].append(clob_latency)
            if len(self.stats['clob_latencies']) > 50:
                self.stats['clob_latencies'] = self.stats['clob_latencies'][-50:]
            
            if result and 'error' not in result:
                status = result.get('status', result.get('orderStatus', 'UNKNOWN')).upper()
                
                # GTC at above-market price should fill immediately as taker
                # MATCHED = filled, LIVE = resting (not what we want)
                filled = status in ['MATCHED', 'FILLED']
                
                if status == 'LIVE':
                    # Order is resting, not filled - try to cancel it
                    order_id = result.get('orderID', result.get('id'))
                    logger.warning(f"[{pos.symbol}] {side} order LIVE (not filled) - may need to cancel {order_id}")
                
                if filled:
                    cost = shares * price
                    
                    # Update position
                    if side == 'YES':
                        pos.yes_shares += shares
                        pos.yes_cost += cost
                    else:
                        pos.no_shares += shares
                        pos.no_cost += cost
                    pos.trades_count += 1
                    
                    self.stats['orders_filled'] += 1
                    self.stats['total_spent'] += cost
                    
                    logger.info(f"[{pos.symbol}] ✓ {side} @ {price:.4f} x {shares:.1f} = ${cost:.2f} | {clob_latency:.0f}ms")
                    self._save_trade(pos, side, price, shares, cost, total_latency, clob_latency, result.get('orderID'))
                    
                    return {'filled': True, 'status': status}
                else:
                    logger.info(f"[{pos.symbol}] ✗ {side} FOK not filled: {status}")
                    return {'filled': False, 'status': status}
            else:
                error = result.get('error', 'Unknown') if result else 'No response'
                logger.warning(f"[{pos.symbol}] ✗ {side} order error: {error}")
                self.stats['orders_failed'] += 1
                return {'filled': False, 'error': error}
                
        except Exception as e:
            logger.error(f"[{pos.symbol}] {side} execution error: {e}")
            self.stats['orders_failed'] += 1
            return {'filled': False, 'error': str(e)}
    
    def _send_fok_order_sync(self, token_id: str, price: float, shares: float, side: str) -> dict:
        """Synchronous order - use GTC for better fill rates"""
        try:
            # Polymarket requirements: amounts must have max 2 decimals
            price = round(price, 2)
            shares = int(shares)
            
            if shares < 1:
                return {'error': 'Shares too small (< 1)'}
            
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=float(shares),
                side=BUY,
            )
            
            signed_order = self.clob.create_order(order_args)
            # Use GTC - will stay open if not immediately filled
            # Better fill rate than FOK
            resp = self.clob.post_order(signed_order, OrderType.GTC)
            
            return resp if resp else {'error': 'No response'}
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _emergency_sell(self, pos: Position, side: str, buy_price: float, shares: float):
        """Emergency sell to close partial position"""
        token_id = pos.yes_token_id if side == 'YES' else pos.no_token_id
        
        # Get current best bid to sell
        book = self.orderbook.get(token_id)
        bids = book.get('bids', [])
        
        if not bids:
            logger.error(f"[{pos.symbol}] No bids to sell {side}!")
            return
        
        sell_price = bids[0]['price']
        loss = (buy_price - sell_price) * shares
        
        logger.warning(f"[{pos.symbol}] Emergency sell {side} @ {sell_price:.4f} (bought @ {buy_price:.4f}, loss: ${loss:.2f})")
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._send_sell_order_sync,
                token_id, sell_price, shares
            )
            
            if result and 'error' not in result:
                status = result.get('status', 'UNKNOWN').upper()
                if status in ['MATCHED', 'FILLED', 'LIVE']:
                    # Revert position
                    if side == 'YES':
                        pos.yes_shares -= shares
                        pos.yes_cost -= shares * buy_price
                    else:
                        pos.no_shares -= shares
                        pos.no_cost -= shares * buy_price
                    
                    logger.info(f"[{pos.symbol}] Emergency sell {status} - position closed")
                else:
                    logger.error(f"[{pos.symbol}] Emergency sell status: {status}")
            else:
                logger.error(f"[{pos.symbol}] Emergency sell failed: {result}")
                
        except Exception as e:
            logger.error(f"[{pos.symbol}] Emergency sell error: {e}")
    
    def _send_sell_order_sync(self, token_id: str, price: float, shares: float) -> dict:
        """Synchronous sell order"""
        from py_clob_client.order_builder.constants import SELL
        try:
            # Use whole number shares
            price = round(price, 2)
            shares = int(shares)
            
            if shares < 1:
                return {'error': 'Shares too small (< 1)'}
            
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=float(shares),
                side=SELL,
            )
            
            signed_order = self.clob.create_order(order_args)
            resp = self.clob.post_order(signed_order, OrderType.FOK)
            
            return resp if resp else {'error': 'No response'}
            
        except Exception as e:
            return {'error': str(e)}
    
    async def execute(self, pos: Position, side: str, price: float, shares: float, recv_time: float):
        """Legacy single-side execution (for simulation)"""
        token_id = pos.yes_token_id if side == 'YES' else pos.no_token_id
        cost = shares * price
        
        if self.config.simulation_mode:
            await self._execute_sim(pos, side, price, shares, cost, recv_time)
        else:
            await self._execute_real(pos, side, price, shares, cost, token_id, recv_time)
    
    async def _execute_sim(self, pos, side, price, shares, cost, recv_time):
        """Simulated execution"""
        latency = (time.time() - recv_time) * 1000
        
        # Update position
        if side == 'YES':
            pos.yes_shares += shares
            pos.yes_cost += cost
        else:
            pos.no_shares += shares
            pos.no_cost += cost
        pos.trades_count += 1
        
        self.stats['orders_sent'] += 1
        self.stats['orders_filled'] += 1
        self.stats['total_spent'] += cost
        self.stats['latencies'].append(latency)
        if len(self.stats['latencies']) > 100:
            self.stats['latencies'] = self.stats['latencies'][-100:]
        
        logger.info(f"[{pos.symbol}] [SIM] ✓ {side} @ {price:.4f} x {shares:.1f} = ${cost:.2f}")
        logger.info(f"[{pos.symbol}] [SIM] Position: UP={pos.yes_shares:.1f}@{pos.avg_yes_price:.3f} DOWN={pos.no_shares:.1f}@{pos.avg_no_price:.3f}")
        logger.info(f"[{pos.symbol}] [SIM] Combined: {pos.avg_combined_price:.4f} | Profit: ${pos.guaranteed_profit:.2f}")
        
        self._save_trade(pos, side, price, shares, cost, latency, 0, None)
    
    async def _execute_real(self, pos, side, price, shares, cost, token_id, recv_time):
        """Real execution via py_clob_client"""
        self.stats['orders_sent'] += 1
        
        clob_start = time.time()
        
        try:
            # Run synchronous CLOB call in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._send_order_sync,
                token_id, price, shares
            )
            
            clob_latency = (time.time() - clob_start) * 1000
            total_latency = (time.time() - recv_time) * 1000
            
            self.stats['clob_latencies'].append(clob_latency)
            if len(self.stats['clob_latencies']) > 50:
                self.stats['clob_latencies'] = self.stats['clob_latencies'][-50:]
            
            self.stats['latencies'].append(total_latency)
            if len(self.stats['latencies']) > 100:
                self.stats['latencies'] = self.stats['latencies'][-100:]
            
            if result and 'error' not in result:
                status = result.get('status', result.get('orderStatus', 'UNKNOWN'))
                order_id = result.get('orderID', result.get('id', 'unknown'))
                
                if status.upper() in ['MATCHED', 'LIVE', 'PENDING', 'SUCCESS', 'ACTIVE']:
                    # Update position
                    if side == 'YES':
                        pos.yes_shares += shares
                        pos.yes_cost += cost
                    else:
                        pos.no_shares += shares
                        pos.no_cost += cost
                    pos.trades_count += 1
                    
                    self.stats['orders_filled'] += 1
                    self.stats['total_spent'] += cost
                    
                    logger.info(f"[{pos.symbol}] ✓ {side} @ {price:.4f} x {shares:.1f} = ${cost:.2f}")
                    logger.info(f"[{pos.symbol}] CLOB: {clob_latency:.0f}ms | Total: {total_latency:.0f}ms | Status: {status}")
                    logger.info(f"[{pos.symbol}] Combined: {pos.avg_combined_price:.4f} | Profit: ${pos.guaranteed_profit:.2f}")
                    
                    self._save_trade(pos, side, price, shares, cost, total_latency, clob_latency, order_id)
                else:
                    self.stats['orders_failed'] += 1
                    logger.warning(f"[{pos.symbol}] Order status: {status} | Response: {result}")
            else:
                self.stats['orders_failed'] += 1
                error = result.get('error', 'Unknown error') if result else 'No response'
                logger.error(f"[{pos.symbol}] Order failed: {error}")
                
        except Exception as e:
            self.stats['orders_failed'] += 1
            logger.error(f"[{pos.symbol}] Execution error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _send_order_sync(self, token_id: str, price: float, shares: float) -> dict:
        """Synchronous order send (called from thread pool)"""
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=shares,
                side=BUY,
            )
            
            signed_order = self.clob.create_order(order_args)
            resp = self.clob.post_order(signed_order, OrderType.GTC)
            
            return resp if resp else {'error': 'No response'}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _save_trade(self, pos, side, price, shares, cost, latency, clob_latency, order_id):
        self.db.execute(
            'INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)',
            (datetime.now(timezone.utc).isoformat(), pos.symbol, side,
             price, shares, cost, latency, clob_latency,
             pos.avg_combined_price, pos.guaranteed_profit, order_id)
        )
        self.db.commit()
    
    # ==================== STATUS ====================
    
    def print_status(self):
        uptime = int(time.time() - self.start_time)
        avg_lat = sum(self.stats['latencies']) / len(self.stats['latencies']) if self.stats['latencies'] else 0
        avg_clob = sum(self.stats['clob_latencies']) / len(self.stats['clob_latencies']) if self.stats['clob_latencies'] else 0
        
        mode = "SIM" if self.config.simulation_mode else "PROD"
        
        logger.info("=" * 70)
        logger.info(f"GABAGOOL ULTRA [{mode}] | Uptime: {uptime//60}m {uptime%60}s")
        logger.info(f"  WSS: {self.stats['wss_messages']} msgs, {self.stats['wss_reconnects']} reconnects")
        logger.info(f"  Opportunities: {self.stats['opportunities']}")
        logger.info(f"  Orders: {self.stats['orders_sent']} sent, {self.stats['orders_filled']} filled, {self.stats['orders_failed']} failed")
        logger.info(f"  Spent: ${self.stats['total_spent']:.2f}")
        logger.info(f"  Latency: {avg_lat:.1f}ms total, {avg_clob:.1f}ms CLOB")
        
        total_profit = sum(p.guaranteed_profit for p in self.positions.values())
        logger.info(f"  Guaranteed profit: ${total_profit:.2f}")
        
        for pos in self.positions.values():
            if pos.trades_count > 0:
                logger.info(f"  [{pos.symbol}] UP={pos.yes_shares:.0f}@{pos.avg_yes_price:.3f} DOWN={pos.no_shares:.0f}@{pos.avg_no_price:.3f} = ${pos.guaranteed_profit:.2f}")
        logger.info("=" * 70)
    
    # ==================== MAIN ====================
    
    async def run(self):
        logger.info("Starting GabagoolUltra...")
        self.running = True
        
        await self.fetch_markets()
        
        tasks = [
            asyncio.create_task(self.wss_loop()),
            asyncio.create_task(self._market_loop()),
            asyncio.create_task(self._status_loop()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            for t in tasks:
                t.cancel()
            self.print_status()
            logger.info("Stopped")
            logger.stop()  # Flush and stop async logger
    
    async def _market_loop(self):
        while self.running:
            await asyncio.sleep(self.config.market_refresh_interval)
            markets_changed = await self.fetch_markets()
            
            if markets_changed and self.ws:
                logger.info("🔄 New 15-min slot - forcing WSS reconnect")
                self.force_reconnect = True
                try:
                    await self.ws.close()
                except:
                    pass
            # No resubscribe needed if markets didn't change - WSS stays connected
    
    async def _status_loop(self):
        while self.running:
            await asyncio.sleep(60)
            self.print_status()

# ========================== MAIN ==========================

def main():
    config = Config.from_env()
    
    logger.info(f"SIMULATION_MODE from env: {os.getenv('SIMULATION_MODE')}")
    logger.info(f"Parsed simulation_mode: {config.simulation_mode}")
    
    if not os.getenv('POLYMARKET_PRIVATE_KEY'):
        logger.error("Set POLYMARKET_PRIVATE_KEY in .env")
        return
    
    try:
        bot = GabagoolUltra(config)
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
