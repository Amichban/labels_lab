# QuantX Data Access Context

## ClickHouse Cloud Connection

### Database Details
- **Database Name**: `quantx` (CRITICAL: Not `default`)
- **Host**: Your ClickHouse Cloud endpoint
- **Port**: 8443 (HTTPS) or 9440 (Native TCP/TLS)
- **Authentication**: Username/password from your ClickHouse Cloud instance

### Connection Example (Python)
```python
from clickhouse_driver import Client

client = Client(
    host='your-clickhouse-cloud-endpoint.clickhouse.cloud',
    port=9440,
    user='your-username',
    password='your-password',
    database='quantx',
    secure=True,
    verify=True,
    settings={'use_numpy': True}
)
```

## Core Data Tables

### 1. Market Snapshots (`quantx.snapshots`)
Raw OHLCV market data with bid/ask spreads.

**Schema**:
```sql
CREATE TABLE quantx.snapshots (
    instrument_id String,
    granularity String,
    ts DateTime64(3),  -- IMPORTANT: Column is 'ts' not 'timestamp'
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    bid Float64,
    ask Float64,
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
ORDER BY (instrument_id, granularity, ts)
PARTITION BY toYYYYMM(ts);
```

**Sample Query**:
```sql
SELECT ts, open, high, low, close, volume
FROM quantx.snapshots
WHERE instrument_id = 'EUR_USD' 
  AND granularity = 'H1'
  AND ts >= '2025-01-01'
ORDER BY ts DESC
LIMIT 100;
```

### 2. Technical Features (`quantx.features`)
Computed technical indicators for each snapshot.

**Schema**:
```sql
CREATE TABLE quantx.features (
    instrument_id String,
    granularity String,
    ts DateTime64(3),
    ema_20 Float64,
    ema_50 Float64,
    ema_200 Float64,
    rsi_14 Float64,
    atr_14 Float64,
    volume_sma_20 Float64,
    volatility_20 Float64,
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
ORDER BY (instrument_id, granularity, ts);
```

### 3. Support/Resistance Levels (`quantx.levels`)
Detected support and resistance levels.

**Schema**:
```sql
CREATE TABLE quantx.levels (
    level_id String,
    instrument_id String,
    granularity String,
    price Float64,
    created_at DateTime64(3),
    initial_type String,  -- 'support' or 'resistance'
    current_type String,
    status String,  -- 'active' or 'inactive'
    deactivated_at Nullable(DateTime64(3)),
    last_event_at DateTime64(3),
    last_event_type String
) ENGINE = MergeTree()
ORDER BY (instrument_id, granularity, created_at);
```

### 4. Level Events (`quantx.level_events`)
Historical events for each level (touches, breaks, flips).

**Schema**:
```sql
CREATE TABLE quantx.level_events (
    event_id String,
    level_id String,
    instrument_id String,
    granularity String,
    ts DateTime64(3),  -- IMPORTANT: 'ts' not 'event_timestamp'
    event_type String,  -- NEW_SUPPORT, FLIP_TO_RESISTANCE, TOUCH_UP, etc.
    level_price Float64,
    candle_open Float64,
    candle_high Float64,
    candle_low Float64,
    candle_close Float64,
    penetration Float64,
    created_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
ORDER BY (instrument_id, granularity, ts);
```

### 5. Level Lifecycle (`quantx.level_lifecycle`)
Denormalized view with complete level history and enhanced metrics.

**Key Columns**:
- `event_timestamps`: Array of all event timestamps
- `event_types`: Array of all event types
- `enhanced_strength`: Calculated strength score (0-10)
- `touch_volumes`: Volume at each touch event
- `touch_rebounds_3candle`: Price movement after touches

## Available Instruments & Granularities

### Instruments (29 FX pairs + indices)
- **Major FX**: EUR_USD, GBP_USD, USD_JPY, USD_CHF, USD_CAD, AUD_USD, NZD_USD
- **Crosses**: EUR_GBP, EUR_JPY, GBP_JPY, EUR_CHF, EUR_CAD, AUD_JPY, etc.
- **Indices**: SPX500_USD, NAS100_USD, US30_USD, etc.

### Granularities
- **H1**: Hourly (most complete data)
- **H4**: 4-hour (timestamps at 1,5,9,13,17,21 UTC)
- **D**: Daily
- **W**: Weekly

### Current Data Coverage (EUR_USD)
- **H1**: 2002-05 to present (~131,000 records)
- **H4**: 2002-05 to present (~32,000 records)
- **Daily**: 2002-05 to present (~5,000 records)
- **Weekly**: 2002-05 to present (~1,200 records)

## Firestore Market Data Source

### Connection
```python
from google.cloud import firestore
import os

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/service-account.json'

db = firestore.Client(project='your-gcp-project-id')
```

### Document Structure
```
Collection: candles/{instrument}/{granularity}/data
Document ID: {timestamp_seconds}
Fields: {
    "o": open_price,
    "h": high_price,
    "l": low_price,
    "c": close_price,
    "v": volume,
    "ts": timestamp,
    "bid": bid_price,
    "ask": ask_price,
    "complete": true/false
}
```

### Real-time Listener Example
```python
def on_snapshot(doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        data = doc.to_dict()
        if data.get('complete', False):
            # Process completed candle
            process_candle(data)

# Listen to EUR_USD H1 candles
collection = db.collection('candles/EUR_USD/H1/data')
query = collection.where('complete', '==', True).order_by('ts', direction=firestore.Query.DESCENDING).limit(1)
query.on_snapshot(on_snapshot)
```

## Redis State Management

### Connection
```python
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
```

### Key Patterns
- Active levels: `levels:{instrument}:{granularity}:active`
- Processing state: `state:{instrument}:{granularity}:indicators`
- Last processed: `last_processed:{instrument}:{granularity}`

## Essential Queries

### Get Latest Price
```sql
SELECT ts, close, bid, ask
FROM quantx.snapshots
WHERE instrument_id = 'EUR_USD' AND granularity = 'H1'
ORDER BY ts DESC
LIMIT 1;
```

### Get Active Levels
```sql
SELECT level_id, price, current_type, created_at, 
       last_event_type, last_event_at
FROM quantx.levels
WHERE instrument_id = 'EUR_USD' 
  AND granularity = 'H1'
  AND status = 'active'
ORDER BY price DESC;
```

### Get Recent Events
```sql
SELECT ts, event_type, level_price, candle_close, penetration
FROM quantx.level_events
WHERE instrument_id = 'EUR_USD' 
  AND granularity = 'H1'
  AND ts >= now() - INTERVAL 7 DAY
ORDER BY ts DESC;
```

### Get Level Strength
```sql
SELECT level_id, price, enhanced_strength,
       length(event_timestamps) as total_events,
       arrayCount(x -> x = 'TOUCHED', event_types) as touch_count
FROM quantx.level_lifecycle
WHERE instrument_id = 'EUR_USD' 
  AND granularity = 'H1'
  AND status = 'active'
ORDER BY enhanced_strength DESC
LIMIT 20;
```

## Environment Variables

```bash
# ClickHouse Cloud
CLICKHOUSE_HOST=your-endpoint.clickhouse.cloud
CLICKHOUSE_PORT=9440
CLICKHOUSE_USER=your-username
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_DATABASE=quantx

# Google Cloud (for Firestore)
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Redis (optional, for state management)
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Python Dependencies

```txt
clickhouse-driver==0.2.6
google-cloud-firestore==2.11.1
redis==4.5.4
pandas==2.0.3
numpy==1.24.3
```

## Key Implementation Notes

1. **Timestamps**: All timestamp columns are named `ts`, not `timestamp` or `event_timestamp`
2. **Database**: Always use `quantx` database, never `default`
3. **Dictionary Access**: ClickHouse client returns dictionaries with column names as keys
4. **H4 Timestamps**: H4 candles occur at 1,5,9,13,17,21 UTC (not 0,4,8,12,16,20)
5. **Complete Candles**: Only process candles where `complete=true` in Firestore
6. **Level IDs**: Format is `{instrument}_{granularity}_{timestamp}_{index}`
7. **Event Types**: NEW_SUPPORT, NEW_RESISTANCE, FLIP_TO_SUPPORT, FLIP_TO_RESISTANCE, TOUCH_UP, TOUCH_DOWN, BREAK_SUPPORT, BREAK_RESISTANCE, DEACTIVATE_UP, DEACTIVATE_DOWN

## Cloud Deployment Considerations

1. **ClickHouse Cloud**: Use secure connection with TLS
2. **Firestore**: Use service account with minimal required permissions
3. **Network**: Ensure your Cloud Run/Functions can access ClickHouse Cloud
4. **Secrets**: Use Secret Manager for credentials
5. **Monitoring**: Set up alerts for data gaps or processing failures

## Contact & Support

- ClickHouse Cloud Dashboard: [Your ClickHouse Cloud URL]
- GCP Project: [Your GCP Project ID]
- Firestore Console: https://console.cloud.google.com/firestore