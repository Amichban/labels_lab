"""
Validation CLI tool for Issue #8

Command-line interface for comprehensive data validation framework.
Provides tools for manual validation, monitoring, and debugging.

Following test-runner guidance for CLI tools:
- Clear, actionable output
- Progress reporting
- Error handling and recovery
- Integration with monitoring systems
"""

import click
import asyncio
import json
import sys
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from src.validation.label_validator import label_validator, ValidationSeverity
from src.validation.validation_metrics import validation_metrics_collector
from src.core.label_computation import computation_engine
from src.models.data_models import Candle, Granularity
from src.utils.timestamp_aligner import TimestampAligner


@click.group()
@click.version_option()
def cli():
    """
    Validation CLI for comprehensive data validation framework (Issue #8)
    
    Provides tools for validating label computations, monitoring validation
    metrics, and debugging data quality issues.
    """
    pass


@cli.command()
@click.option('--instrument-id', required=True, help='Instrument ID (e.g., EUR/USD)')
@click.option('--granularity', required=True, type=click.Choice(['M1', 'M5', 'M15', 'H1', 'H4', 'D', 'W']), 
              help='Time granularity')
@click.option('--timestamp', required=True, help='Timestamp (ISO format: 2024-01-15T09:00:00)')
@click.option('--open', required=True, type=float, help='Open price')
@click.option('--high', required=True, type=float, help='High price')
@click.option('--low', required=True, type=float, help='Low price')
@click.option('--close', required=True, type=float, help='Close price')
@click.option('--volume', required=True, type=float, help='Volume')
@click.option('--horizon-periods', default=6, type=int, help='Horizon periods for validation')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json']), 
              help='Output format')
def validate_candle(instrument_id, granularity, timestamp, open, high, low, close, volume, 
                   horizon_periods, output_format):
    """
    Validate a single candle for data quality issues
    
    Example:
        validation-cli validate-candle --instrument-id EUR/USD --granularity H4 \\
            --timestamp 2024-01-15T09:00:00 --open 1.0500 --high 1.0580 \\
            --low 1.0450 --close 1.0520 --volume 1000
    """
    try:
        # Parse timestamp
        ts = datetime.fromisoformat(timestamp)
        
        # Create candle
        candle = Candle(
            instrument_id=instrument_id,
            granularity=Granularity(granularity),
            ts=ts,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
        
        # Run pre-computation validation
        result = label_validator.validate_pre_computation(candle, horizon_periods)
        
        # Output results
        if output_format == 'json':
            output_data = {
                "candle": {
                    "instrument_id": instrument_id,
                    "granularity": granularity,
                    "timestamp": timestamp,
                    "ohlcv": {"open": open, "high": high, "low": low, "close": close, "volume": volume}
                },
                "validation": {
                    "is_valid": result.is_valid,
                    "total_issues": len(result.issues),
                    "validation_time_ms": result.validation_time_ms,
                    "issues": [
                        {
                            "category": issue.category.value,
                            "severity": issue.severity.value,
                            "message": issue.message,
                            "details": issue.details
                        }
                        for issue in result.issues
                    ]
                }
            }
            click.echo(json.dumps(output_data, indent=2))
        else:
            # Text output
            click.echo(f"🔍 Validating candle: {instrument_id} {granularity} {timestamp}")
            click.echo(f"📊 OHLCV: O={open} H={high} L={low} C={close} V={volume}")
            click.echo()
            
            if result.is_valid:
                click.echo(click.style("✅ Validation PASSED", fg='green', bold=True))
            else:
                click.echo(click.style("❌ Validation FAILED", fg='red', bold=True))
            
            click.echo(f"⏱️  Validation time: {result.validation_time_ms:.2f}ms")
            click.echo(f"🔢 Total issues: {len(result.issues)}")
            click.echo()
            
            if result.issues:
                click.echo("📋 Issues found:")
                for i, issue in enumerate(result.issues, 1):
                    severity_color = {
                        'critical': 'red',
                        'error': 'red',
                        'warning': 'yellow',
                        'info': 'blue'
                    }.get(issue.severity.value, 'white')
                    
                    click.echo(f"  {i}. [{click.style(issue.severity.value.upper(), fg=severity_color)}] "
                              f"{issue.category.value}: {issue.message}")
                    if issue.details:
                        for key, value in issue.details.items():
                            click.echo(f"     └─ {key}: {value}")
                click.echo()
        
        sys.exit(0 if result.is_valid else 1)
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option('--instrument-id', required=True, help='Instrument ID')
@click.option('--granularity', required=True, type=click.Choice(['M1', 'M5', 'M15', 'H1', 'H4', 'D', 'W']))
@click.option('--timestamp', required=True, help='Timestamp (ISO format)')
@click.option('--horizon-periods', default=6, type=int, help='Horizon periods')
@click.option('--label-types', default='enhanced_triple_barrier,vol_scaled_return,mfe_mae', 
              help='Comma-separated label types to compute')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json']))
def compute_and_validate(instrument_id, granularity, timestamp, horizon_periods, label_types, output_format):
    """
    Compute labels and validate the results
    
    Example:
        validation-cli compute-and-validate --instrument-id EUR/USD --granularity H4 \\
            --timestamp 2024-01-15T09:00:00 --label-types enhanced_triple_barrier
    """
    async def _compute_and_validate():
        try:
            # Parse inputs
            ts = datetime.fromisoformat(timestamp)
            label_type_list = [lt.strip() for lt in label_types.split(',')]
            
            # Create a basic candle (would normally come from data source)
            candle = Candle(
                instrument_id=instrument_id,
                granularity=Granularity(granularity),
                ts=ts,
                open=1.0500,  # Mock data
                high=1.0580,
                low=1.0450,
                close=1.0520,
                volume=1000.0,
                atr_14=0.0045
            )
            
            # Compute labels with validation enabled
            label_set = await computation_engine.compute_labels(
                candle=candle,
                horizon_periods=horizon_periods,
                label_types=label_type_list,
                use_cache=False
            )
            
            # Get validation stats
            validation_stats = computation_engine.get_validation_stats()
            
            if output_format == 'json':
                output_data = {
                    "computation": {
                        "instrument_id": instrument_id,
                        "granularity": granularity,
                        "timestamp": timestamp,
                        "label_types": label_type_list,
                        "computation_time_ms": label_set.computation_time_ms
                    },
                    "labels": {
                        "enhanced_triple_barrier": (
                            {
                                "label": label_set.enhanced_triple_barrier.label,
                                "barrier_hit": label_set.enhanced_triple_barrier.barrier_hit.value,
                                "time_to_barrier": label_set.enhanced_triple_barrier.time_to_barrier,
                                "level_adjusted": label_set.enhanced_triple_barrier.level_adjusted
                            } 
                            if label_set.enhanced_triple_barrier else None
                        ),
                        "forward_return": label_set.forward_return,
                        "vol_scaled_return": label_set.vol_scaled_return,
                        "mfe": label_set.mfe,
                        "mae": label_set.mae,
                        "profit_factor": label_set.profit_factor
                    },
                    "validation_stats": validation_stats
                }
                click.echo(json.dumps(output_data, indent=2))
            else:
                # Text output
                click.echo(f"🧮 Computing labels: {instrument_id} {granularity} {timestamp}")
                click.echo(f"📋 Label types: {', '.join(label_type_list)}")
                click.echo(f"⏱️  Computation time: {label_set.computation_time_ms:.2f}ms")
                click.echo()
                
                # Display computed labels
                if label_set.enhanced_triple_barrier:
                    etb = label_set.enhanced_triple_barrier
                    click.echo("🎯 Enhanced Triple Barrier:")
                    click.echo(f"   Label: {etb.label}")
                    click.echo(f"   Barrier hit: {etb.barrier_hit.value}")
                    click.echo(f"   Time to barrier: {etb.time_to_barrier}")
                    click.echo(f"   Level adjusted: {etb.level_adjusted}")
                    click.echo()
                
                if label_set.forward_return is not None:
                    click.echo(f"📈 Forward return: {label_set.forward_return:.6f}")
                
                if label_set.vol_scaled_return is not None:
                    click.echo(f"📊 Vol-scaled return: {label_set.vol_scaled_return:.6f}")
                
                if label_set.mfe is not None and label_set.mae is not None:
                    click.echo(f"📐 MFE: {label_set.mfe:.6f}")
                    click.echo(f"📐 MAE: {label_set.mae:.6f}")
                    if label_set.profit_factor:
                        click.echo(f"📐 Profit Factor: {label_set.profit_factor:.6f}")
                
                click.echo()
                click.echo("🔍 Validation Statistics:")
                click.echo(f"   Total computations: {validation_stats['total_computations']}")
                click.echo(f"   Pre-validation failures: {validation_stats['pre_validation_failures']}")
                click.echo(f"   Post-validation failures: {validation_stats['post_validation_failures']}")
                
                if validation_stats['total_computations'] > 0:
                    pre_rate = validation_stats.get('pre_validation_failure_rate', 0) * 100
                    post_rate = validation_stats.get('post_validation_failure_rate', 0) * 100
                    click.echo(f"   Pre-validation failure rate: {pre_rate:.1f}%")
                    click.echo(f"   Post-validation failure rate: {post_rate:.1f}%")
        
        except Exception as e:
            click.echo(click.style(f"❌ Error: {str(e)}", fg='red'), err=True)
            sys.exit(1)
    
    asyncio.run(_compute_and_validate())


@cli.command()
@click.option('--window-minutes', type=int, help='Time window for metrics (default: all-time)')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json', 'prometheus']))
def metrics(window_minutes, output_format):
    """
    Display validation metrics and health status
    
    Example:
        validation-cli metrics --window-minutes 60
    """
    try:
        if output_format == 'prometheus':
            # Export in Prometheus format
            prometheus_metrics = validation_metrics_collector.export_metrics('prometheus')
            click.echo(prometheus_metrics)
            return
        
        # Get metrics summary
        summary = validation_metrics_collector.get_metrics_summary(window_minutes)
        health_score, health_breakdown = validation_metrics_collector.get_health_score()
        active_alerts = validation_metrics_collector.get_active_alerts()
        
        if output_format == 'json':
            output_data = {
                "metrics_summary": summary,
                "health_score": health_score,
                "health_breakdown": health_breakdown,
                "active_alerts": active_alerts
            }
            click.echo(json.dumps(output_data, indent=2))
        else:
            # Text output
            window_desc = f"last {window_minutes} minutes" if window_minutes else "all time"
            click.echo(f"📊 Validation Metrics ({window_desc})")
            click.echo("=" * 50)
            
            # Overall stats
            click.echo(f"Total validations: {summary['total_validations']}")
            click.echo(f"Successful: {summary['successful_validations']}")
            click.echo(f"Failed: {summary['failed_validations']}")
            
            if summary['total_validations'] > 0:
                click.echo(f"Success rate: {summary['success_rate']:.1%}")
                click.echo(f"Failure rate: {summary['failure_rate']:.1%}")
            
            click.echo()
            
            # Health score
            health_color = 'green' if health_score >= 80 else ('yellow' if health_score >= 60 else 'red')
            click.echo(click.style(f"🏥 Health Score: {health_score:.1f}/100", fg=health_color, bold=True))
            click.echo()
            
            # Issue breakdown by severity
            if summary['severity_counts']:
                click.echo("🚨 Issues by Severity:")
                for severity, count in summary['severity_counts'].items():
                    severity_color = {
                        'critical': 'red',
                        'error': 'red', 
                        'warning': 'yellow',
                        'info': 'blue'
                    }.get(severity, 'white')
                    click.echo(f"   {severity.upper()}: {click.style(str(count), fg=severity_color)}")
                click.echo()
            
            # Issue breakdown by category
            if summary['category_counts']:
                click.echo("📋 Issues by Category:")
                for category, count in summary['category_counts'].items():
                    click.echo(f"   {category.replace('_', ' ').title()}: {count}")
                click.echo()
            
            # Timing stats
            if summary['timing_stats']:
                timing = summary['timing_stats']
                click.echo("⏱️  Performance Stats:")
                click.echo(f"   Average: {timing['avg_ms']:.2f}ms")
                click.echo(f"   Median: {timing['median_ms']:.2f}ms")
                click.echo(f"   95th percentile: {timing['p95_ms']:.2f}ms")
                click.echo(f"   Max: {timing['max_ms']:.2f}ms")
                click.echo()
            
            # Active alerts
            if active_alerts:
                click.echo(click.style("🚨 Active Alerts:", fg='red', bold=True))
                for alert in active_alerts:
                    severity_color = {
                        'low': 'blue',
                        'medium': 'yellow',
                        'high': 'red',
                        'critical': 'red'
                    }.get(alert['severity'], 'white')
                    
                    click.echo(f"   [{click.style(alert['severity'].upper(), fg=severity_color)}] "
                              f"{alert['rule_name']}: {alert['description']}")
                click.echo()
            else:
                click.echo(click.style("✅ No active alerts", fg='green'))
    
    except Exception as e:
        click.echo(click.style(f"❌ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option('--top-n', default=10, type=int, help='Number of top issues to show')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json']))
def analyze_failures(top_n, output_format):
    """
    Analyze validation failure patterns
    
    Example:
        validation-cli analyze-failures --top-n 5
    """
    try:
        analysis = validation_metrics_collector.get_failure_analysis(top_n)
        
        if output_format == 'json':
            click.echo(json.dumps(analysis, indent=2))
        else:
            click.echo("🔍 Validation Failure Analysis")
            click.echo("=" * 40)
            
            if analysis['total_failures'] == 0:
                click.echo(click.style("✅ No validation failures recorded", fg='green'))
                return
            
            click.echo(f"Total failures: {analysis['total_failures']}")
            click.echo(f"Failure rate: {analysis['failure_rate']:.1%}")
            click.echo()
            
            if analysis['top_issue_patterns']:
                click.echo(f"🔝 Top {len(analysis['top_issue_patterns'])} Issue Patterns:")
                for i, (pattern, count) in enumerate(analysis['top_issue_patterns'], 1):
                    category, severity = pattern.split(':')
                    severity_color = {
                        'critical': 'red',
                        'error': 'red',
                        'warning': 'yellow', 
                        'info': 'blue'
                    }.get(severity, 'white')
                    
                    click.echo(f"   {i}. {category.replace('_', ' ').title()} "
                              f"[{click.style(severity.upper(), fg=severity_color)}]: {count} occurrences")
                click.echo()
            
            # Category-severity matrix
            if analysis['category_severity_matrix']:
                click.echo("📊 Issue Distribution Matrix:")
                for category, severities in analysis['category_severity_matrix'].items():
                    click.echo(f"   {category.replace('_', ' ').title()}:")
                    for severity, count in severities.items():
                        severity_color = {
                            'critical': 'red',
                            'error': 'red',
                            'warning': 'yellow',
                            'info': 'blue'
                        }.get(severity, 'white')
                        click.echo(f"     {click.style(severity.upper(), fg=severity_color)}: {count}")
    
    except Exception as e:
        click.echo(click.style(f"❌ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option('--hours', default=24, type=int, help='Hours of history to analyze')
@click.option('--output-format', default='text', type=click.Choice(['text', 'json']))
def trends(hours, output_format):
    """
    Show historical validation trends
    
    Example:
        validation-cli trends --hours 48
    """
    try:
        trends_data = validation_metrics_collector.get_historical_trends(hours)
        
        if output_format == 'json':
            click.echo(json.dumps(trends_data, indent=2))
        else:
            if trends_data.get('status') == 'insufficient_data':
                click.echo(click.style(f"⚠️  Insufficient data for {hours} hour analysis", fg='yellow'))
                return
            
            click.echo(f"📈 Validation Trends ({hours} hours)")
            click.echo("=" * 40)
            click.echo(f"Data points analyzed: {trends_data['data_points']}")
            click.echo()
            
            # Failure rate trend
            failure_trend = trends_data['failure_rate_trend']
            trend_emoji = "📈" if failure_trend['trend'] == 'increasing' else "📊"
            click.echo(f"{trend_emoji} Failure Rate Trend:")
            click.echo(f"   Current: {failure_trend['current']:.1%}")
            click.echo(f"   Average: {failure_trend['average']:.1%}")
            click.echo(f"   Range: {failure_trend['min']:.1%} - {failure_trend['max']:.1%}")
            click.echo(f"   Trend: {failure_trend['trend']}")
            click.echo()
            
            # Critical issues trend
            critical_trend = trends_data['critical_issues_trend']
            click.echo("🚨 Critical Issues Trend:")
            click.echo(f"   Current: {critical_trend['current']}")
            click.echo(f"   Total: {critical_trend['total']}")
            click.echo(f"   Peak: {critical_trend['peak']}")
            click.echo()
            
            # Error issues trend
            error_trend = trends_data['error_issues_trend']
            click.echo("❌ Error Issues Trend:")
            click.echo(f"   Current: {error_trend['current']}")
            click.echo(f"   Total: {error_trend['total']}")
            click.echo(f"   Peak: {error_trend['peak']}")
    
    except Exception as e:
        click.echo(click.style(f"❌ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option('--granularity', required=True, type=click.Choice(['H1', 'H4', 'D']))
@click.option('--count', default=5, type=int, help='Number of test timestamps to generate')
def test_alignment(granularity, count):
    """
    Test timestamp alignment for given granularity
    
    Example:
        validation-cli test-alignment --granularity H4 --count 10
    """
    try:
        click.echo(f"🕐 Testing timestamp alignment for {granularity}")
        click.echo("=" * 50)
        
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        
        for i in range(count):
            # Generate test timestamp (may not be aligned)
            if granularity == 'H1':
                test_time = base_time + timedelta(hours=i, minutes=15)  # Add minutes offset
            elif granularity == 'H4':
                test_time = base_time + timedelta(hours=4*i, minutes=30)  # Add minutes offset
            else:  # D
                test_time = base_time + timedelta(days=i, hours=6)  # Add hours offset
            
            # Check alignment
            aligned_time = TimestampAligner.align_to_granularity(test_time, granularity)
            is_aligned = TimestampAligner.validate_alignment(test_time, granularity)
            
            status_icon = "✅" if is_aligned else "❌"
            click.echo(f"{status_icon} {test_time.isoformat()} → {aligned_time.isoformat()} "
                      f"({'ALIGNED' if is_aligned else 'NOT ALIGNED'})")
        
        if granularity == 'H4':
            click.echo()
            click.echo("ℹ️  H4 candles should align to: 01:00, 05:00, 09:00, 13:00, 17:00, 21:00 UTC")
    
    except Exception as e:
        click.echo(click.style(f"❌ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show validation framework version info"""
    click.echo("🔍 Comprehensive Validation Framework v1.0.0")
    click.echo("Issue #8 Implementation")
    click.echo()
    click.echo("Features:")
    click.echo("  ✅ Look-ahead bias detection")
    click.echo("  ✅ Data consistency validation")
    click.echo("  ✅ Timestamp alignment checks") 
    click.echo("  ✅ Statistical distribution testing")
    click.echo("  ✅ Path granularity validation")
    click.echo("  ✅ Real-time metrics and alerting")
    click.echo("  ✅ Performance monitoring")


if __name__ == '__main__':
    cli()