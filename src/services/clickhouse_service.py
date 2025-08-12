"""
ClickHouse connection and query service with circuit breaker protection

Enhanced with resilience patterns including:
- Circuit breaker protection
- Automatic retry with exponential backoff  
- Fallback handlers for graceful degradation
- Health monitoring and recovery

Issue #14: Circuit breakers and failover mechanisms
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from contextlib import contextmanager
from clickhouse_driver import Client
from config.settings import settings
from .circuit_breaker import CircuitBreakerConfig, with_retry
from .resilience_manager import get_resilience_manager
from .fallback_handlers import get_fallback_handler

logger = logging.getLogger(__name__)


class ClickHouseService:
    """Service for ClickHouse database operations with resilience features"""
    
    def __init__(self):
        """Initialize ClickHouse connection with circuit breaker"""
        self.connection_params = {
            'host': settings.clickhouse_host,
            'port': settings.clickhouse_port,
            'user': settings.clickhouse_user,
            'password': settings.clickhouse_password,
            'database': settings.clickhouse_database,
            'secure': settings.clickhouse_secure,
            'verify': settings.clickhouse_secure,
            'settings': {
                'use_numpy': True,
                'max_block_size': 10000
            }
        }
        self._client: Optional[Client] = None
        
        # Get circuit breaker from resilience manager
        self.resilience_manager = get_resilience_manager()
        self.fallback_handler = get_fallback_handler('clickhouse')
        
        # Register service if not already registered
        try:
            self.circuit_breaker = self.resilience_manager.get_circuit_breaker('clickhouse')
            if not self.circuit_breaker:
                # Register with resilience manager
                circuit_config = CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=120.0,
                    timeout=30.0,
                    success_threshold=2,
                    expected_exception=(Exception,)
                )
                self.circuit_breaker = self.resilience_manager.register_service(
                    'clickhouse', 
                    'database', 
                    'critical', 
                    circuit_config,
                    self._fallback_query_handler
                )
        except Exception as e:
            logger.warning(f"Could not register ClickHouse with resilience manager: {e}")
            self.circuit_breaker = None
            
        logger.info("ClickHouse service initialized with circuit breaker protection")
    
    @property
    def client(self) -> Client:
        """Get or create ClickHouse client"""
        if self._client is None:
            self._client = Client(**self.connection_params)
            logger.info(f"Connected to ClickHouse at {settings.clickhouse_host}:{settings.clickhouse_port}")
        return self._client
    
    @contextmanager
    def get_client(self):
        """Context manager for ClickHouse client"""
        client = Client(**self.connection_params)
        try:
            yield client
        finally:
            client.disconnect()
    
    def _fallback_query_handler(self, query: str, params: Optional[Dict] = None, operation_type: str = "read") -> List[Dict[str, Any]]:
        """Fallback handler for ClickHouse query failures"""
        if self.fallback_handler:
            import asyncio
            # Convert sync call to async for fallback handler
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to handle this differently
                logger.warning("Fallback handler called from sync context, returning empty result")
                return []
            else:
                return loop.run_until_complete(
                    self.fallback_handler.handle_query_failure(query, params, operation_type)
                )
        return []
    
    @with_retry(max_attempts=3, base_delay=1.0, max_delay=30.0, 
                exceptions=(Exception,))
    def execute(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as list of dicts with circuit breaker protection
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of dictionaries with query results
        """
        # Use circuit breaker if available
        if self.circuit_breaker:
            return self.circuit_breaker.call(self._execute_query, query, params)
        else:
            # Fallback to direct execution
            return self._execute_query(query, params)
    
    def _execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Internal query execution method"""
        try:
            with self.get_client() as client:
                result = client.execute(query, params or {}, with_column_types=True)
                if not result:
                    return []
                
                data, columns = result[0], result[1]
                column_names = [col[0] for col in columns]
                
                results = [dict(zip(column_names, row)) for row in data]
                
                # Cache successful query result for fallback
                if self.fallback_handler:
                    self.fallback_handler.cache_successful_query(query, params, results)
                
                return results
                
        except Exception as e:
            logger.error(f"ClickHouse query failed: {e}")
            # Signal service degradation to resilience manager
            if hasattr(self.resilience_manager, 'fallback_orchestrator'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            self.resilience_manager.fallback_orchestrator.service_degraded('clickhouse')
                        )
                except:
                    pass
            raise
    
    @with_retry(max_attempts=2, base_delay=2.0, max_delay=10.0,
                exceptions=(Exception,))
    def insert(self, table: str, data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert data into table with circuit breaker protection
        
        Args:
            table: Table name (e.g., 'quantx.labels')
            data: List of dictionaries to insert
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        # Use circuit breaker if available
        if self.circuit_breaker:
            return self.circuit_breaker.call(self._insert_data, table, data)
        else:
            return self._insert_data(table, data)
    
    def _insert_data(self, table: str, data: List[Dict[str, Any]]) -> int:
        """Internal insert method"""
        try:
            with self.get_client() as client:
                client.execute(f'INSERT INTO {table} VALUES', data)
                logger.info(f"Inserted {len(data)} rows into {table}")
                return len(data)
        except Exception as e:
            logger.error(f"ClickHouse insert failed: {e}")
            # Signal service degradation 
            if hasattr(self.resilience_manager, 'fallback_orchestrator'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            self.resilience_manager.fallback_orchestrator.service_degraded('clickhouse')
                        )
                except:
                    pass
            raise
    
    def fetch_snapshots(self,
                       instrument_id: str,
                       granularity: str,
                       start_ts: datetime,
                       end_ts: datetime) -> List[Dict[str, Any]]:
        """
        Fetch market snapshots for given parameters
        
        Args:
            instrument_id: Instrument identifier (e.g., 'EUR_USD')
            granularity: Time granularity (e.g., 'H1', 'H4')
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            List of snapshot dictionaries
        """
        query = """
        SELECT 
            instrument_id, granularity, ts,
            open, high, low, close, volume,
            bid, ask,
            ema_20, ema_50, ema_200,
            rsi_14, atr_14,
            volume_sma_20, volatility_20
        FROM quantx.snapshots
        LEFT JOIN quantx.features USING (instrument_id, granularity, ts)
        WHERE instrument_id = %(instrument_id)s
          AND granularity = %(granularity)s
          AND ts >= %(start_ts)s
          AND ts < %(end_ts)s
        ORDER BY ts
        """
        
        params = {
            'instrument_id': instrument_id,
            'granularity': granularity,
            'start_ts': start_ts.isoformat(),
            'end_ts': end_ts.isoformat()
        }
        
        return self.execute(query, params)
    
    def fetch_active_levels(self,
                           instrument_id: str,
                           granularity: str,
                           at_timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Fetch active support/resistance levels at given timestamp
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            at_timestamp: Point in time to check
            
        Returns:
            List of active level dictionaries
        """
        query = """
        SELECT 
            level_id, instrument_id, granularity,
            price, created_at, current_type, status,
            last_event_type, last_event_at
        FROM quantx.levels
        WHERE instrument_id = %(instrument_id)s
          AND granularity = %(granularity)s
          AND created_at <= %(at_ts)s
          AND (status = 'active' 
               OR (status = 'inactive' AND deactivated_at > %(at_ts)s))
        ORDER BY price
        """
        
        params = {
            'instrument_id': instrument_id,
            'granularity': granularity,
            'at_ts': at_timestamp.isoformat()
        }
        
        return self.execute(query, params)
    
    def fetch_level_events(self,
                          instrument_id: str,
                          granularity: str,
                          start_ts: datetime,
                          end_ts: datetime) -> List[Dict[str, Any]]:
        """
        Fetch level events in time range
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            List of level event dictionaries
        """
        query = """
        SELECT 
            event_id, level_id, instrument_id, granularity,
            ts, event_type, level_price,
            candle_open, candle_high, candle_low, candle_close,
            penetration
        FROM quantx.level_events
        WHERE instrument_id = %(instrument_id)s
          AND granularity = %(granularity)s
          AND ts >= %(start_ts)s
          AND ts < %(end_ts)s
        ORDER BY ts, level_id
        """
        
        params = {
            'instrument_id': instrument_id,
            'granularity': granularity,
            'start_ts': start_ts.isoformat(),
            'end_ts': end_ts.isoformat()
        }
        
        return self.execute(query, params)
    
    def check_connection(self) -> bool:
        """
        Check if ClickHouse connection is working
        
        Returns:
            True if connection is successful
        """
        try:
            # Use direct connection check without circuit breaker to avoid recursive calls
            result = self._execute_query("SELECT 1 as check")
            is_healthy = len(result) > 0 and result[0]['check'] == 1
            
            # Notify resilience manager of successful health check
            if is_healthy and hasattr(self.resilience_manager, 'fallback_orchestrator'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            self.resilience_manager.fallback_orchestrator.service_recovered('clickhouse')
                        )
                except:
                    pass
            
            return is_healthy
        except Exception as e:
            logger.error(f"ClickHouse connection check failed: {e}")
            return False
    
    def get_latest_snapshot_time(self,
                                instrument_id: str,
                                granularity: str) -> Optional[datetime]:
        """
        Get the timestamp of the latest snapshot for an instrument
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            
        Returns:
            Latest timestamp or None if no data
        """
        query = """
        SELECT max(ts) as latest_ts
        FROM quantx.snapshots
        WHERE instrument_id = %(instrument_id)s
          AND granularity = %(granularity)s
        """
        
        params = {
            'instrument_id': instrument_id,
            'granularity': granularity
        }
        
        result = self.execute(query, params)
        if result and result[0]['latest_ts']:
            return result[0]['latest_ts']
        return None
    
    def close(self):
        """Close the ClickHouse connection"""
        if self._client:
            self._client.disconnect()
            self._client = None
            logger.info("ClickHouse connection closed")


# Global service instance
clickhouse_service = ClickHouseService()