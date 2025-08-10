#!/usr/bin/env python3
"""
Example script demonstrating the Priority Labels implementation for Issue #6.

This script shows how to use the 5 priority labels:
- Label 2: Volatility-Scaled Returns
- Labels 9-10: MFE/MAE with Profit Factor
- Label 12: Level Retouch Count
- Label 16: Breakout Beyond Level
- Label 17: Flip Within Horizon
"""

import asyncio
from datetime import datetime, timedelta
from src.core.label_computation import LabelComputationEngine
from src.models.data_models import Candle, Granularity


async def demo_priority_labels():
    """Demonstrate the priority labels computation."""
    
    print("🚀 Priority Labels Demo (Issue #6)")
    print("=" * 50)
    
    # Create a sample candle
    sample_candle = Candle(
        instrument_id="EUR/USD",
        granularity=Granularity.H4,
        ts=datetime(2024, 1, 15, 13, 0, 0),  # H4 candle at 13:00 UTC
        open=1.0500,
        high=1.0580,
        low=1.0450,
        close=1.0520,
        volume=1500.0,
        atr_14=0.0045  # 0.45% ATR
    )
    
    print(f"📊 Sample Candle: {sample_candle.instrument_id} {sample_candle.granularity.value}")
    print(f"   Timestamp: {sample_candle.ts}")
    print(f"   OHLC: {sample_candle.open:.4f} / {sample_candle.high:.4f} / {sample_candle.low:.4f} / {sample_candle.close:.4f}")
    print(f"   ATR: {sample_candle.atr_14:.4f}")
    print()
    
    # Initialize computation engine
    engine = LabelComputationEngine()
    
    # Define priority labels to compute
    priority_labels = [
        "vol_scaled_return",      # Label 2
        "mfe_mae",                # Labels 9-10
        "level_retouch_count",    # Label 12
        "breakout_beyond_level",  # Label 16
        "flip_within_horizon"     # Label 17
    ]
    
    print("🔮 Computing Priority Labels...")
    print(f"   Labels: {priority_labels}")
    print(f"   Horizon: 6 periods (24 hours for H4)")
    print()
    
    try:
        # Compute labels (would normally connect to ClickHouse/Redis)
        label_set = await engine.compute_labels(
            candle=sample_candle,
            horizon_periods=6,
            label_types=priority_labels,
            use_cache=False  # Disable cache for demo
        )
        
        # Display results
        print("✅ Priority Labels Results:")
        print("-" * 30)
        
        print(f"Label 2 - Volatility-Scaled Return:")
        if label_set.vol_scaled_return is not None:
            print(f"   Value: {label_set.vol_scaled_return:.4f}")
            print(f"   Formula: (P_{{t+H}} - P_t) / ATR_t")
        else:
            print(f"   Value: None (no future data available)")
        print()
        
        print(f"Labels 9-10 - MFE/MAE with Profit Factor:")
        print(f"   MFE: {label_set.mfe:.4f} pip" if label_set.mfe else "   MFE: None")
        print(f"   MAE: {label_set.mae:.4f} pip" if label_set.mae else "   MAE: None") 
        if label_set.profit_factor:
            print(f"   Profit Factor: {label_set.profit_factor:.4f}")
            print(f"   Formula: MFE / |MAE|")
        else:
            print(f"   Profit Factor: None")
        print()
        
        print(f"Label 12 - Level Retouch Count:")
        if label_set.retouch_count is not None:
            print(f"   Count: {label_set.retouch_count} touches")
            print(f"   Threshold: 0.1% touch tolerance, 0.2% break threshold")
        else:
            print(f"   Count: None (no active levels)")
        print()
        
        print(f"Label 16 - Breakout Beyond Level:")
        if label_set.breakout_occurred is not None:
            print(f"   Breakout: {'Yes' if label_set.breakout_occurred else 'No'}")
            print(f"   Threshold: 0.2% beyond level price")
        else:
            print(f"   Breakout: None (no active levels)")
        print()
        
        print(f"Label 17 - Flip Within Horizon:")
        if label_set.flip_occurred is not None:
            print(f"   Flip: {'Yes' if label_set.flip_occurred else 'No'}")
            print(f"   Type: Support ↔ Resistance conversion")
        else:
            print(f"   Flip: None (no active levels)")
        print()
        
        # Metadata
        print("📈 Computation Metadata:")
        print(f"   Computation Time: {label_set.computation_time_ms:.2f}ms")
        print(f"   Label Version: {label_set.label_version}")
        print(f"   Computed At: {label_set.computed_at}")
        
    except Exception as e:
        print(f"❌ Error computing labels: {e}")
        print("Note: This demo requires ClickHouse and Redis services to be running.")
        print("In production, the engine would fetch real market data and level information.")


def demo_formula_explanations():
    """Explain the mathematical formulas for each priority label."""
    
    print("\n📚 Priority Labels Formula Reference")
    print("=" * 50)
    
    print("Label 2: Volatility-Scaled Returns")
    print("   Formula: (P_{t+H} - P_t) / ATR_t")
    print("   Where:")
    print("     P_t     = Current price (close)")
    print("     P_{t+H} = Price at horizon end")
    print("     ATR_t   = Average True Range at time t")
    print("   Purpose: Normalize returns by volatility for better comparison")
    print()
    
    print("Labels 9-10: MFE/MAE with Profit Factor")
    print("   MFE = max(P_{t+τ} - P_t) for τ ∈ [0, H]")
    print("   MAE = min(P_{t+τ} - P_t) for τ ∈ [0, H]")
    print("   Profit Factor = MFE / |MAE|")
    print("   Where:")
    print("     τ = Time steps within horizon")
    print("     H = Horizon length")
    print("   Purpose: Measure maximum favorable/adverse price excursions")
    print()
    
    print("Label 12: Level Retouch Count")
    print("   Count touches at active S/R levels within horizon")
    print("   Touch: |price - level| / level ≤ 0.1%")
    print("   Exclusion: Breakout distance > 0.2%")
    print("   Purpose: Quantify level strength and price interaction")
    print()
    
    print("Label 16: Breakout Beyond Level")
    print("   Boolean: Did price break significantly beyond level?")
    print("   Threshold: |price - level| / level > 0.2%")
    print("   Purpose: Identify significant level violations")
    print()
    
    print("Label 17: Flip Within Horizon")
    print("   Boolean: Did level flip from support ↔ resistance?")
    print("   Conditions:")
    print("     1. Price breaks level (>0.2%)")
    print("     2. Price retests level as opposite type (≤0.1%)")
    print("   Purpose: Detect dynamic level role changes")
    print()


async def main():
    """Main demo function."""
    await demo_priority_labels()
    demo_formula_explanations()


if __name__ == "__main__":
    asyncio.run(main())