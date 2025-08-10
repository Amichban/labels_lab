"""
ClickHouse connection and query service
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from contextlib import contextmanager
from clickhouse_driver import Client
from config.settings import settings

logger = logging.getLogger(__name__)


class ClickHouseService:
    """Service for ClickHouse database operations"""
    
    def __init__(self):
        """Initialize ClickHouse connection"""
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
    
    def execute(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as list of dicts
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of dictionaries with query results
        """
        try:
            with self.get_client() as client:
                result = client.execute(query, params or {}, with_column_types=True)
                if not result:
                    return []
                
                data, columns = result[0], result[1]
                column_names = [col[0] for col in columns]
                
                return [dict(zip(column_names, row)) for row in data]
        except Exception as e:
            logger.error(f"ClickHouse query failed: {e}")
            raise
    
    def insert(self, table: str, data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert data into table
        
        Args:
            table: Table name (e.g., 'quantx.labels')
            data: List of dictionaries to insert
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
        
        try:
            with self.get_client() as client:
                client.execute(f'INSERT INTO {table} VALUES', data)
                logger.info(f"Inserted {len(data)} rows into {table}")
                return len(data)
        except Exception as e:
            logger.error(f"ClickHouse insert failed: {e}")
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
            result = self.execute("SELECT 1 as check")
            return len(result) > 0 and result[0]['check'] == 1
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