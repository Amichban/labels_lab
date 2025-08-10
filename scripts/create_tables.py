#!/usr/bin/env python3
"""
Create ClickHouse tables for label computation system
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from clickhouse_driver import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_tables():
    """Create necessary tables in ClickHouse"""
    
    # Connect to ClickHouse
    client = Client(
        host=os.environ['CLICKHOUSE_HOST'],
        port=int(os.environ.get('CLICKHOUSE_PORT', 9440)),
        user=os.environ['CLICKHOUSE_USER'],
        password=os.environ['CLICKHOUSE_PASSWORD'],
        database=os.environ.get('CLICKHOUSE_DATABASE', 'quantx'),
        secure=True,
        verify=True
    )
    
    print("🔗 Connected to ClickHouse")
    
    # Create labels table
    labels_table = """
    CREATE TABLE IF NOT EXISTS quantx.labels (
        instrument_id String,
        granularity String,
        ts DateTime64(3),
        
        -- Label 11.a: Enhanced Triple Barrier
        enhanced_triple_barrier_label Int8,
        enhanced_triple_barrier_time UInt16,
        enhanced_triple_barrier_price Float64,
        enhanced_triple_barrier_adjusted Bool,
        
        -- Core labels
        forward_return Float64,
        vol_scaled_return Float64,
        return_sign Int8,
        return_quantile UInt8,
        
        -- Path metrics
        mfe Float64,
        mae Float64,
        profit_factor Float64,
        max_penetration Float64,
        
        -- Level-specific
        retouch_count UInt8,
        next_touch_time UInt16,
        breakout_occurred Bool,
        flip_occurred Bool,
        nearest_level_distance Float64,
        
        -- Risk metrics
        drawdown_depth Float64,
        time_underwater UInt16,
        path_skewness Float64,
        
        -- Metadata
        label_version String,
        computed_at DateTime64(3) DEFAULT now64(3),
        
        INDEX idx_enhanced_barrier enhanced_triple_barrier_label TYPE minmax GRANULARITY 1,
        INDEX idx_return_quantile return_quantile TYPE minmax GRANULARITY 1
    ) ENGINE = MergeTree()
    ORDER BY (instrument_id, granularity, ts)
    PARTITION BY toYYYYMM(ts)
    SETTINGS index_granularity = 8192
    """
    
    try:
        client.execute(labels_table)
        print("✅ Created/verified quantx.labels table")
    except Exception as e:
        print(f"❌ Error creating labels table: {e}")
    
    # Create level_labels table
    level_labels_table = """
    CREATE TABLE IF NOT EXISTS quantx.level_labels (
        level_id String,
        instrument_id String,
        granularity String,
        ts DateTime64(3),
        
        -- Level interaction labels
        distance_to_level Float64,
        penetration_depth Float64,
        time_at_level UInt16,
        consecutive_touches UInt8,
        
        -- Forward looking
        next_event_type String,
        time_to_next_event UInt16,
        level_holds Bool,
        
        computed_at DateTime64(3) DEFAULT now64(3)
    ) ENGINE = MergeTree()
    ORDER BY (level_id, ts)
    PARTITION BY toYYYYMM(ts)
    """
    
    try:
        client.execute(level_labels_table)
        print("✅ Created/verified quantx.level_labels table")
    except Exception as e:
        print(f"❌ Error creating level_labels table: {e}")
    
    # Verify tables exist
    tables = client.execute("SHOW TABLES FROM quantx")
    print(f"\n📊 Tables in quantx database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Check if we have the source tables we need
    required_tables = ['snapshots', 'features', 'levels', 'level_events']
    for table in required_tables:
        result = client.execute(f"EXISTS TABLE quantx.{table}")
        if result[0][0]:
            count = client.execute(f"SELECT count() FROM quantx.{table}")[0][0]
            print(f"✅ Source table quantx.{table} exists with {count:,} rows")
        else:
            print(f"⚠️  Source table quantx.{table} does not exist")
    
    print("\n✅ Table setup complete!")

if __name__ == "__main__":
    try:
        create_tables()
    except KeyError as e:
        print(f"❌ Missing environment variable: {e}")
        print("Please ensure these are set: CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)