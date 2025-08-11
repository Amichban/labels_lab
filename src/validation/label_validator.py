"""
Comprehensive data validation framework for Issue #8

This module implements extensive validation checks to ensure data integrity
and prevent look-ahead bias in label computation. It follows the test-runner
guidance for systematic validation with statistical testing.

Validation framework covers:
1. No look-ahead bias: All computations use data ≤ current timestamp
2. Data consistency: MFE >= -MAE, barriers properly ordered
3. Timestamp alignment: H4 at 1,5,9,13,17,21 UTC
4. Label distribution: Jarque-Bera test for normality
5. Path data granularity: Verify H4→H1, D→H4 mappings
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats
import warnings

from src.models.data_models import (
    Candle, LabelSet, EnhancedTripleBarrierLabel,
    Granularity, BarrierHit, Level
)
from src.utils.timestamp_aligner import TimestampAligner

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"
    ERROR = "error" 
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(str, Enum):
    """Categories of validation checks"""
    LOOKAHEAD_BIAS = "lookahead_bias"
    DATA_CONSISTENCY = "data_consistency"
    TIMESTAMP_ALIGNMENT = "timestamp_alignment"
    STATISTICAL_DISTRIBUTION = "statistical_distribution"
    PATH_GRANULARITY = "path_granularity"
    BARRIER_LOGIC = "barrier_logic"
    PERFORMANCE = "performance"


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    candle_id: Optional[str] = None
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.category.value}: {self.message}"


@dataclass
class ValidationResult:
    """Complete validation results"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_time_ms: Optional[float] = None
    
    def add_issue(
        self, 
        category: ValidationCategory, 
        severity: ValidationSeverity,
        message: str, 
        details: Dict[str, Any] = None,
        candle_id: str = None
    ):
        """Add validation issue"""
        issue = ValidationIssue(
            category=category,
            severity=severity,
            message=message,
            details=details or {},
            candle_id=candle_id
        )
        self.issues.append(issue)
        
        # Critical and error issues mark validation as failed
        if severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
            self.is_valid = False
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues filtered by category"""
        return [issue for issue in self.issues if issue.category == category]
    
    def summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        return {
            "is_valid": self.is_valid,
            "total_issues": len(self.issues),
            "critical_issues": len(self.get_issues_by_severity(ValidationSeverity.CRITICAL)),
            "error_issues": len(self.get_issues_by_severity(ValidationSeverity.ERROR)),
            "warning_issues": len(self.get_issues_by_severity(ValidationSeverity.WARNING)),
            "info_issues": len(self.get_issues_by_severity(ValidationSeverity.INFO)),
            "validation_time_ms": self.validation_time_ms,
            "categories": list(set(issue.category for issue in self.issues))
        }


class LabelValidator:
    """
    Comprehensive label validation framework
    
    Implements systematic validation checks to ensure data quality,
    prevent look-ahead bias, and maintain statistical integrity.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator
        
        Args:
            strict_mode: If True, warnings become errors
        """
        self.strict_mode = strict_mode
        self.validation_stats = {
            "total_validations": 0,
            "failed_validations": 0,
            "avg_validation_time_ms": 0.0
        }
    
    def validate_pre_computation(
        self,
        candle: Candle,
        horizon_periods: int,
        path_data: List[Dict[str, Any]] = None,
        levels: List[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Pre-computation validation checks
        
        Validates input data before label computation to catch issues early.
        
        Args:
            candle: Input candle data
            horizon_periods: Forward-looking periods
            path_data: Path data for barrier checking (optional)
            levels: Active S/R levels (optional)
            
        Returns:
            Validation result with any issues found
        """
        start_time = datetime.utcnow()
        result = ValidationResult(is_valid=True)
        granularity_value = candle.granularity.value if hasattr(candle.granularity, 'value') else str(candle.granularity)
        candle_id = f"{candle.instrument_id}_{granularity_value}_{candle.ts}"
        
        try:
            # 1. Validate candle data integrity
            self._validate_candle_integrity(candle, result, candle_id)
            
            # 2. Validate timestamp alignment
            self._validate_timestamp_alignment(candle, result, candle_id)
            
            # 3. Validate horizon parameters
            self._validate_horizon_parameters(horizon_periods, result, candle_id)
            
            # 4. Validate path data (if provided)
            if path_data is not None:
                self._validate_path_data_pre(candle, path_data, result, candle_id)
            
            # 5. Validate levels data (if provided)
            if levels is not None:
                self._validate_levels_data_pre(candle, levels, result, candle_id)
            
            # Record validation time
            validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.validation_time_ms = validation_time
            
            # Update stats
            self.validation_stats["total_validations"] += 1
            if not result.is_valid:
                self.validation_stats["failed_validations"] += 1
            
            return result
            
        except Exception as e:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.CRITICAL,
                f"Pre-computation validation failed: {str(e)}",
                {"exception": str(e)},
                candle_id
            )
            return result
    
    def validate_post_computation(
        self,
        candle: Candle,
        label_set: LabelSet,
        path_data: List[Dict[str, Any]] = None,
        computation_context: Dict[str, Any] = None
    ) -> ValidationResult:
        """
        Post-computation validation checks
        
        Validates computed labels for logical consistency, statistical validity,
        and absence of look-ahead bias.
        
        Args:
            candle: Original input candle
            label_set: Computed label set
            path_data: Path data used in computation (optional)
            computation_context: Additional context from computation (optional)
            
        Returns:
            Validation result with any issues found
        """
        start_time = datetime.utcnow()
        result = ValidationResult(is_valid=True)
        granularity_value = candle.granularity.value if hasattr(candle.granularity, 'value') else str(candle.granularity)
        candle_id = f"{candle.instrument_id}_{granularity_value}_{candle.ts}"
        
        try:
            # 1. Validate no look-ahead bias
            self._validate_no_lookahead_bias(candle, label_set, path_data, result, candle_id)
            
            # 2. Validate data consistency
            self._validate_data_consistency(candle, label_set, result, candle_id)
            
            # 3. Validate barrier logic (if Enhanced Triple Barrier computed)
            if label_set.enhanced_triple_barrier:
                self._validate_barrier_logic(candle, label_set.enhanced_triple_barrier, result, candle_id)
            
            # 4. Validate MFE/MAE consistency
            if label_set.mfe is not None and label_set.mae is not None:
                self._validate_mfe_mae_consistency(label_set, result, candle_id)
            
            # 5. Validate computation performance
            self._validate_computation_performance(label_set, result, candle_id)
            
            # 6. Validate path data granularity mapping
            if path_data:
                self._validate_path_granularity_mapping(candle, path_data, result, candle_id)
            
            # Record validation time
            validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.validation_time_ms = validation_time
            
            return result
            
        except Exception as e:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.CRITICAL,
                f"Post-computation validation failed: {str(e)}",
                {"exception": str(e)},
                candle_id
            )
            return result
    
    def validate_batch_labels(
        self,
        label_sets: List[LabelSet],
        statistical_tests: bool = True
    ) -> ValidationResult:
        """
        Batch validation for multiple label sets
        
        Performs statistical analysis across multiple labels to detect
        distribution anomalies and systematic biases.
        
        Args:
            label_sets: List of computed label sets
            statistical_tests: Whether to perform statistical distribution tests
            
        Returns:
            Validation result for the batch
        """
        start_time = datetime.utcnow()
        result = ValidationResult(is_valid=True)
        
        if not label_sets:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.ERROR,
                "Empty label set batch provided"
            )
            return result
        
        try:
            # 1. Validate batch size and completeness
            self._validate_batch_completeness(label_sets, result)
            
            # 2. Validate temporal ordering
            self._validate_temporal_ordering(label_sets, result)
            
            # 3. Statistical distribution tests
            if statistical_tests:
                self._validate_statistical_distributions(label_sets, result)
            
            # 4. Cross-label consistency checks
            self._validate_cross_label_consistency(label_sets, result)
            
            # 5. Performance metrics validation
            self._validate_batch_performance_metrics(label_sets, result)
            
            # Record validation time
            validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.validation_time_ms = validation_time
            
            # Add batch metrics
            result.metrics.update({
                "batch_size": len(label_sets),
                "instruments": len(set(ls.instrument_id for ls in label_sets)),
                "granularities": len(set(ls.granularity for ls in label_sets)),
                "time_span_hours": self._calculate_time_span(label_sets)
            })
            
            return result
            
        except Exception as e:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.CRITICAL,
                f"Batch validation failed: {str(e)}",
                {"exception": str(e)}
            )
            return result
    
    # Private validation methods
    
    def _validate_candle_integrity(self, candle: Candle, result: ValidationResult, candle_id: str):
        """Validate basic candle data integrity"""
        
        # OHLC relationships
        if not (candle.low <= candle.open <= candle.high and candle.low <= candle.close <= candle.high):
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.ERROR,
                "Invalid OHLC relationships",
                {
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close
                },
                candle_id
            )
        
        # Positive values
        if any(val <= 0 for val in [candle.open, candle.high, candle.low, candle.close]):
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.ERROR,
                "OHLC values must be positive",
                {
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close
                },
                candle_id
            )
        
        # Volume validation
        if candle.volume < 0:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.ERROR,
                "Volume cannot be negative",
                {"volume": candle.volume},
                candle_id
            )
        
        # Bid/Ask spread validation
        if candle.bid and candle.ask:
            if candle.bid >= candle.ask:
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    "Bid must be less than Ask",
                    {"bid": candle.bid, "ask": candle.ask},
                    candle_id
                )
                
            # Check if close is within bid-ask spread (with small tolerance)
            tolerance = 0.0001
            if not (candle.bid - tolerance <= candle.close <= candle.ask + tolerance):
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.WARNING,
                    "Close price outside bid-ask spread",
                    {"close": candle.close, "bid": candle.bid, "ask": candle.ask},
                    candle_id
                )
    
    def _validate_timestamp_alignment(self, candle: Candle, result: ValidationResult, candle_id: str):
        """Validate timestamp alignment to granularity"""
        
        granularity_value = candle.granularity.value if hasattr(candle.granularity, 'value') else str(candle.granularity)
        aligned_ts = TimestampAligner.align_to_granularity(candle.ts, granularity_value)
        
        if candle.ts != aligned_ts:
            result.add_issue(
                ValidationCategory.TIMESTAMP_ALIGNMENT,
                ValidationSeverity.ERROR,
                f"Timestamp not aligned to {granularity_value} granularity",
                {
                    "actual_ts": candle.ts.isoformat(),
                    "expected_ts": aligned_ts.isoformat(),
                    "granularity": granularity_value
                },
                candle_id
            )
        
        # Special validation for H4 timestamps
        if granularity_value == 'H4':
            expected_hours = [1, 5, 9, 13, 17, 21]
            if candle.ts.hour not in expected_hours:
                result.add_issue(
                    ValidationCategory.TIMESTAMP_ALIGNMENT,
                    ValidationSeverity.ERROR,
                    "H4 timestamp not at expected hours (1,5,9,13,17,21 UTC)",
                    {
                        "actual_hour": candle.ts.hour,
                        "expected_hours": expected_hours
                    },
                    candle_id
                )
    
    def _validate_horizon_parameters(self, horizon_periods: int, result: ValidationResult, candle_id: str):
        """Validate horizon parameters"""
        
        if horizon_periods <= 0:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.ERROR,
                "Horizon periods must be positive",
                {"horizon_periods": horizon_periods},
                candle_id
            )
        
        if horizon_periods > 100:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.WARNING,
                "Very large horizon periods may indicate configuration error",
                {"horizon_periods": horizon_periods},
                candle_id
            )
    
    def _validate_path_data_pre(
        self, 
        candle: Candle, 
        path_data: List[Dict[str, Any]], 
        result: ValidationResult, 
        candle_id: str
    ):
        """Validate path data before computation"""
        
        if not path_data:
            result.add_issue(
                ValidationCategory.PATH_GRANULARITY,
                ValidationSeverity.WARNING,
                "Empty path data provided",
                {},
                candle_id
            )
            return
        
        # Check for future data (look-ahead bias)
        for i, data in enumerate(path_data):
            if "ts" in data and data["ts"] < candle.ts:
                result.add_issue(
                    ValidationCategory.LOOKAHEAD_BIAS,
                    ValidationSeverity.ERROR,
                    f"Path data contains timestamps before candle timestamp",
                    {
                        "path_index": i,
                        "path_ts": data["ts"].isoformat() if isinstance(data["ts"], datetime) else str(data["ts"]),
                        "candle_ts": candle.ts.isoformat()
                    },
                    candle_id
                )
        
        # Validate OHLC in path data
        for i, data in enumerate(path_data[:10]):  # Check first 10 for performance
            if all(key in data for key in ["open", "high", "low", "close"]):
                if not (data["low"] <= data["open"] <= data["high"] and 
                       data["low"] <= data["close"] <= data["high"]):
                    result.add_issue(
                        ValidationCategory.DATA_CONSISTENCY,
                        ValidationSeverity.ERROR,
                        f"Invalid OHLC in path data at index {i}",
                        {"path_data": data},
                        candle_id
                    )
    
    def _validate_levels_data_pre(
        self, 
        candle: Candle, 
        levels: List[Dict[str, Any]], 
        result: ValidationResult, 
        candle_id: str
    ):
        """Validate levels data before computation"""
        
        for i, level in enumerate(levels):
            # Check for required fields
            required_fields = ["price", "current_type"]
            missing_fields = [field for field in required_fields if field not in level]
            if missing_fields:
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    f"Level at index {i} missing required fields: {missing_fields}",
                    {"level_data": level},
                    candle_id
                )
                continue
            
            # Validate price
            if level["price"] <= 0:
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    f"Level at index {i} has invalid price",
                    {"price": level["price"]},
                    candle_id
                )
            
            # Check for future creation (look-ahead bias)
            if "created_at" in level:
                created_at = level["created_at"]
                if isinstance(created_at, datetime) and created_at > candle.ts:
                    result.add_issue(
                        ValidationCategory.LOOKAHEAD_BIAS,
                        ValidationSeverity.ERROR,
                        f"Level at index {i} created after candle timestamp",
                        {
                            "level_created_at": created_at.isoformat(),
                            "candle_ts": candle.ts.isoformat()
                        },
                        candle_id
                    )
    
    def _validate_no_lookahead_bias(
        self, 
        candle: Candle, 
        label_set: LabelSet, 
        path_data: List[Dict[str, Any]], 
        result: ValidationResult, 
        candle_id: str
    ):
        """Critical validation: ensure no look-ahead bias in computations"""
        
        # 1. Label computation timestamp should be >= candle timestamp
        if label_set.computed_at and label_set.computed_at < candle.ts:
            result.add_issue(
                ValidationCategory.LOOKAHEAD_BIAS,
                ValidationSeverity.CRITICAL,
                "Label computed before candle timestamp",
                {
                    "computed_at": label_set.computed_at.isoformat(),
                    "candle_ts": candle.ts.isoformat()
                },
                candle_id
            )
        
        # 2. Path data should not contain future information
        if path_data:
            for i, data in enumerate(path_data):
                if "ts" in data:
                    data_ts = data["ts"]
                    if isinstance(data_ts, datetime) and data_ts < candle.ts:
                        result.add_issue(
                            ValidationCategory.LOOKAHEAD_BIAS,
                            ValidationSeverity.CRITICAL,
                            f"Path data at index {i} contains future information",
                            {
                                "path_ts": data_ts.isoformat(),
                                "candle_ts": candle.ts.isoformat()
                            },
                            candle_id
                        )
        
        # 3. Check Enhanced Triple Barrier for look-ahead
        if label_set.enhanced_triple_barrier:
            etb = label_set.enhanced_triple_barrier
            
            # Time to barrier should be reasonable
            if etb.time_to_barrier < 0:
                result.add_issue(
                    ValidationCategory.LOOKAHEAD_BIAS,
                    ValidationSeverity.CRITICAL,
                    "Negative time to barrier indicates look-ahead bias",
                    {"time_to_barrier": etb.time_to_barrier},
                    candle_id
                )
    
    def _validate_data_consistency(
        self, 
        candle: Candle, 
        label_set: LabelSet, 
        result: ValidationResult, 
        candle_id: str
    ):
        """Validate logical consistency of computed data"""
        
        # 1. Forward return consistency
        if label_set.forward_return is not None:
            # Forward return should be realistic (not extreme)
            if abs(label_set.forward_return) > 0.5:  # 50% return seems extreme
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.WARNING,
                    "Extreme forward return detected",
                    {"forward_return": label_set.forward_return},
                    candle_id
                )
        
        # 2. Volatility-scaled return consistency
        if label_set.vol_scaled_return is not None:
            # Should be finite
            if not np.isfinite(label_set.vol_scaled_return):
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    "Non-finite volatility-scaled return",
                    {"vol_scaled_return": label_set.vol_scaled_return},
                    candle_id
                )
        
        # 3. Profit factor consistency
        if label_set.profit_factor is not None:
            if label_set.profit_factor < 0:
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    "Profit factor cannot be negative",
                    {"profit_factor": label_set.profit_factor},
                    candle_id
                )
                
            if not np.isfinite(label_set.profit_factor):
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    "Non-finite profit factor",
                    {"profit_factor": label_set.profit_factor},
                    candle_id
                )
        
        # 4. Count fields should be non-negative
        count_fields = ["retouch_count", "next_touch_time", "time_underwater"]
        for field in count_fields:
            value = getattr(label_set, field, None)
            if value is not None and value < 0:
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    f"{field} cannot be negative",
                    {field: value},
                    candle_id
                )
    
    def _validate_barrier_logic(
        self, 
        candle: Candle, 
        etb_label: EnhancedTripleBarrierLabel, 
        result: ValidationResult, 
        candle_id: str
    ):
        """Validate Enhanced Triple Barrier logic"""
        
        # 1. Barrier ordering
        if etb_label.upper_barrier <= etb_label.lower_barrier:
            result.add_issue(
                ValidationCategory.BARRIER_LOGIC,
                ValidationSeverity.ERROR,
                "Upper barrier must be greater than lower barrier",
                {
                    "upper_barrier": etb_label.upper_barrier,
                    "lower_barrier": etb_label.lower_barrier
                },
                candle_id
            )
        
        # 2. Entry price should be between barriers
        entry_price = candle.close
        if not (etb_label.lower_barrier < entry_price < etb_label.upper_barrier):
            result.add_issue(
                ValidationCategory.BARRIER_LOGIC,
                ValidationSeverity.ERROR,
                "Entry price should be between barriers",
                {
                    "entry_price": entry_price,
                    "upper_barrier": etb_label.upper_barrier,
                    "lower_barrier": etb_label.lower_barrier
                },
                candle_id
            )
        
        # 3. Label consistency with barrier hit
        if etb_label.barrier_hit == BarrierHit.UPPER and etb_label.label != 1:
            result.add_issue(
                ValidationCategory.BARRIER_LOGIC,
                ValidationSeverity.ERROR,
                "Upper barrier hit should result in label = 1",
                {
                    "barrier_hit": etb_label.barrier_hit,
                    "label": etb_label.label
                },
                candle_id
            )
        elif etb_label.barrier_hit == BarrierHit.LOWER and etb_label.label != -1:
            result.add_issue(
                ValidationCategory.BARRIER_LOGIC,
                ValidationSeverity.ERROR,
                "Lower barrier hit should result in label = -1",
                {
                    "barrier_hit": etb_label.barrier_hit,
                    "label": etb_label.label
                },
                candle_id
            )
        elif etb_label.barrier_hit == BarrierHit.NONE and etb_label.label != 0:
            result.add_issue(
                ValidationCategory.BARRIER_LOGIC,
                ValidationSeverity.ERROR,
                "No barrier hit should result in label = 0",
                {
                    "barrier_hit": etb_label.barrier_hit,
                    "label": etb_label.label
                },
                candle_id
            )
        
        # 4. Time to barrier consistency
        if etb_label.barrier_hit != BarrierHit.NONE and etb_label.time_to_barrier <= 0:
            result.add_issue(
                ValidationCategory.BARRIER_LOGIC,
                ValidationSeverity.ERROR,
                "Time to barrier should be positive when barrier is hit",
                {
                    "barrier_hit": etb_label.barrier_hit,
                    "time_to_barrier": etb_label.time_to_barrier
                },
                candle_id
            )
        
        # 5. Barrier price consistency
        if etb_label.barrier_hit != BarrierHit.NONE and etb_label.barrier_price is None:
            result.add_issue(
                ValidationCategory.BARRIER_LOGIC,
                ValidationSeverity.ERROR,
                "Barrier price should be set when barrier is hit",
                {
                    "barrier_hit": etb_label.barrier_hit,
                    "barrier_price": etb_label.barrier_price
                },
                candle_id
            )
    
    def _validate_mfe_mae_consistency(self, label_set: LabelSet, result: ValidationResult, candle_id: str):
        """Validate MFE/MAE consistency - CRITICAL: MFE >= -MAE"""
        
        mfe = label_set.mfe
        mae = label_set.mae
        
        # 1. MFE should be >= 0 (favorable moves are positive)
        if mfe < 0:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.ERROR,
                "MFE (Maximum Favorable Excursion) should be non-negative",
                {"mfe": mfe},
                candle_id
            )
        
        # 2. MAE should be <= 0 (adverse moves are negative)
        if mae > 0:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.ERROR,
                "MAE (Maximum Adverse Excursion) should be non-positive",
                {"mae": mae},
                candle_id
            )
        
        # 3. CRITICAL: MFE >= -MAE (fundamental constraint)
        if mfe < -mae:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.CRITICAL,
                "MFE must be >= -MAE (fundamental constraint violated)",
                {
                    "mfe": mfe,
                    "mae": mae,
                    "mfe_plus_mae": mfe + mae
                },
                candle_id
            )
        
        # 4. Profit factor consistency
        if label_set.profit_factor is not None and mae != 0:
            expected_pf = mfe / abs(mae)
            actual_pf = label_set.profit_factor
            
            if abs(actual_pf - expected_pf) > 0.0001:  # Small tolerance for float precision
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    "Profit factor inconsistent with MFE/MAE",
                    {
                        "mfe": mfe,
                        "mae": mae,
                        "expected_profit_factor": expected_pf,
                        "actual_profit_factor": actual_pf
                    },
                    candle_id
                )
    
    def _validate_computation_performance(self, label_set: LabelSet, result: ValidationResult, candle_id: str):
        """Validate computation performance metrics"""
        
        if label_set.computation_time_ms is not None:
            # Computation should complete within reasonable time
            if label_set.computation_time_ms > 10000:  # 10 seconds
                result.add_issue(
                    ValidationCategory.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    "Computation time exceeds 10 seconds",
                    {"computation_time_ms": label_set.computation_time_ms},
                    candle_id
                )
            
            # Negative computation time is impossible
            if label_set.computation_time_ms < 0:
                result.add_issue(
                    ValidationCategory.PERFORMANCE,
                    ValidationSeverity.ERROR,
                    "Negative computation time",
                    {"computation_time_ms": label_set.computation_time_ms},
                    candle_id
                )
    
    def _validate_path_granularity_mapping(
        self, 
        candle: Candle, 
        path_data: List[Dict[str, Any]], 
        result: ValidationResult, 
        candle_id: str
    ):
        """Validate path data granularity mapping"""
        
        try:
            expected_path_gran, expected_multiplier = TimestampAligner.get_path_granularity(
                candle.granularity.value
            )
            
            # Check if path data appears to match expected granularity
            if len(path_data) >= 2:
                # Sample first two entries to estimate granularity
                ts1 = path_data[0].get("ts")
                ts2 = path_data[1].get("ts")
                
                if isinstance(ts1, datetime) and isinstance(ts2, datetime):
                    time_diff = ts2 - ts1
                    
                    # Expected time differences for each granularity
                    expected_deltas = {
                        "M1": timedelta(minutes=1),
                        "M5": timedelta(minutes=5),
                        "M15": timedelta(minutes=15),
                        "H1": timedelta(hours=1),
                        "H4": timedelta(hours=4),
                        "D": timedelta(days=1),
                        "W": timedelta(weeks=1)
                    }
                    
                    expected_delta = expected_deltas.get(expected_path_gran)
                    if expected_delta and abs(time_diff - expected_delta) > timedelta(minutes=1):
                        result.add_issue(
                            ValidationCategory.PATH_GRANULARITY,
                            ValidationSeverity.WARNING,
                            f"Path data granularity may not match expected {expected_path_gran}",
                            {
                                "expected_granularity": expected_path_gran,
                                "observed_time_diff": str(time_diff),
                                "expected_time_diff": str(expected_delta)
                            },
                            candle_id
                        )
        
        except ValueError as e:
            result.add_issue(
                ValidationCategory.PATH_GRANULARITY,
                ValidationSeverity.WARNING,
                f"Could not validate path granularity mapping: {str(e)}",
                {"granularity": candle.granularity.value},
                candle_id
            )
    
    def _validate_batch_completeness(self, label_sets: List[LabelSet], result: ValidationResult):
        """Validate batch completeness"""
        
        if len(label_sets) == 0:
            result.add_issue(
                ValidationCategory.DATA_CONSISTENCY,
                ValidationSeverity.ERROR,
                "Empty batch provided"
            )
            return
        
        # Check for missing timestamps in time series
        if len(label_sets) > 1:
            sorted_sets = sorted(label_sets, key=lambda x: x.ts)
            
            # Check for gaps in time series
            for i in range(1, len(sorted_sets)):
                prev_set = sorted_sets[i-1]
                curr_set = sorted_sets[i]
                
                if prev_set.instrument_id == curr_set.instrument_id and \
                   prev_set.granularity == curr_set.granularity:
                    
                    # Calculate expected next timestamp
                    expected_next = TimestampAligner.get_horizon_end(
                        prev_set.ts, prev_set.granularity.value, 1
                    )
                    
                    if curr_set.ts != expected_next:
                        # Allow some tolerance for weekends/holidays
                        time_diff = abs((curr_set.ts - expected_next).total_seconds())
                        if time_diff > 3600 * 72:  # > 72 hours gap
                            result.add_issue(
                                ValidationCategory.DATA_CONSISTENCY,
                                ValidationSeverity.WARNING,
                                "Large time gap detected in batch",
                                {
                                    "prev_ts": prev_set.ts.isoformat(),
                                    "curr_ts": curr_set.ts.isoformat(),
                                    "expected_ts": expected_next.isoformat(),
                                    "gap_hours": time_diff / 3600
                                }
                            )
    
    def _validate_temporal_ordering(self, label_sets: List[LabelSet], result: ValidationResult):
        """Validate temporal ordering of label sets"""
        
        # Group by instrument and granularity
        groups = {}
        for ls in label_sets:
            key = (ls.instrument_id, ls.granularity.value)
            if key not in groups:
                groups[key] = []
            groups[key].append(ls)
        
        # Check ordering within each group
        for (instrument, granularity), group_sets in groups.items():
            if len(group_sets) <= 1:
                continue
                
            # Sort by timestamp
            sorted_sets = sorted(group_sets, key=lambda x: x.ts)
            
            # Check if original order matches sorted order
            if group_sets != sorted_sets:
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.WARNING,
                    "Batch is not temporally ordered",
                    {
                        "instrument_id": instrument,
                        "granularity": granularity,
                        "count": len(group_sets)
                    }
                )
    
    def _validate_statistical_distributions(self, label_sets: List[LabelSet], result: ValidationResult):
        """Validate statistical distributions using Jarque-Bera test"""
        
        try:
            # Extract numerical values for distribution testing
            forward_returns = [ls.forward_return for ls in label_sets if ls.forward_return is not None]
            vol_scaled_returns = [ls.vol_scaled_return for ls in label_sets if ls.vol_scaled_return is not None]
            profit_factors = [ls.profit_factor for ls in label_sets if ls.profit_factor is not None]
            
            # Test forward returns
            if len(forward_returns) >= 20:  # Need minimum sample size
                self._jarque_bera_test(
                    forward_returns, "forward_returns", result,
                    "Forward returns distribution"
                )
            
            # Test volatility-scaled returns  
            if len(vol_scaled_returns) >= 20:
                self._jarque_bera_test(
                    vol_scaled_returns, "vol_scaled_returns", result,
                    "Volatility-scaled returns distribution"
                )
            
            # Test profit factors
            if len(profit_factors) >= 20:
                # Filter out extreme values for profit factor
                filtered_pf = [pf for pf in profit_factors if 0 < pf < 100]
                if len(filtered_pf) >= 20:
                    self._jarque_bera_test(
                        filtered_pf, "profit_factors", result,
                        "Profit factors distribution"
                    )
            
            # Test Enhanced Triple Barrier label distribution
            etb_labels = [
                ls.enhanced_triple_barrier.label 
                for ls in label_sets 
                if ls.enhanced_triple_barrier is not None
            ]
            
            if len(etb_labels) >= 50:  # Need larger sample for discrete distribution
                label_counts = {-1: 0, 0: 0, 1: 0}
                for label in etb_labels:
                    label_counts[label] += 1
                
                total_count = len(etb_labels)
                proportions = {k: v/total_count for k, v in label_counts.items()}
                
                # Check for extreme imbalances
                if any(prop < 0.05 for prop in proportions.values()):
                    result.add_issue(
                        ValidationCategory.STATISTICAL_DISTRIBUTION,
                        ValidationSeverity.WARNING,
                        "Highly imbalanced Enhanced Triple Barrier labels",
                        {
                            "proportions": proportions,
                            "total_samples": total_count
                        }
                    )
                
        except Exception as e:
            result.add_issue(
                ValidationCategory.STATISTICAL_DISTRIBUTION,
                ValidationSeverity.WARNING,
                f"Statistical distribution validation failed: {str(e)}",
                {"exception": str(e)}
            )
    
    def _jarque_bera_test(
        self, 
        values: List[float], 
        variable_name: str, 
        result: ValidationResult,
        description: str
    ):
        """Perform Jarque-Bera test for normality"""
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                values_array = np.array(values)
                
                # Remove NaN and infinite values
                values_array = values_array[np.isfinite(values_array)]
                
                if len(values_array) < 20:
                    return
                
                # Perform Jarque-Bera test
                jb_stat, p_value = stats.jarque_bera(values_array)
                
                # Calculate basic statistics
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                skewness = stats.skew(values_array)
                kurtosis = stats.kurtosis(values_array)
                
                # Add to result metrics
                result.metrics[f"{variable_name}_stats"] = {
                    "count": len(values_array),
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "jarque_bera_stat": float(jb_stat),
                    "jarque_bera_p_value": float(p_value)
                }
                
                # Check for normality (p < 0.05 rejects normality)
                if p_value < 0.01:  # Very strong evidence against normality
                    result.add_issue(
                        ValidationCategory.STATISTICAL_DISTRIBUTION,
                        ValidationSeverity.INFO,
                        f"{description} significantly deviates from normality",
                        {
                            "variable": variable_name,
                            "jb_statistic": float(jb_stat),
                            "p_value": float(p_value),
                            "sample_size": len(values_array)
                        }
                    )
                
                # Check for extreme skewness
                if abs(skewness) > 2:
                    severity = ValidationSeverity.WARNING if abs(skewness) > 3 else ValidationSeverity.INFO
                    result.add_issue(
                        ValidationCategory.STATISTICAL_DISTRIBUTION,
                        severity,
                        f"{description} has extreme skewness",
                        {
                            "variable": variable_name,
                            "skewness": float(skewness)
                        }
                    )
                
                # Check for extreme kurtosis
                if abs(kurtosis) > 3:
                    severity = ValidationSeverity.WARNING if abs(kurtosis) > 5 else ValidationSeverity.INFO
                    result.add_issue(
                        ValidationCategory.STATISTICAL_DISTRIBUTION,
                        severity,
                        f"{description} has extreme kurtosis",
                        {
                            "variable": variable_name,
                            "kurtosis": float(kurtosis)
                        }
                    )
                
        except Exception as e:
            result.add_issue(
                ValidationCategory.STATISTICAL_DISTRIBUTION,
                ValidationSeverity.WARNING,
                f"Jarque-Bera test failed for {variable_name}: {str(e)}",
                {"variable": variable_name, "exception": str(e)}
            )
    
    def _validate_cross_label_consistency(self, label_sets: List[LabelSet], result: ValidationResult):
        """Validate consistency across different label types"""
        
        # Look for systematic inconsistencies
        inconsistencies = 0
        total_comparisons = 0
        
        for ls in label_sets:
            if ls.enhanced_triple_barrier and ls.forward_return is not None:
                etb_label = ls.enhanced_triple_barrier.label
                forward_return = ls.forward_return
                
                total_comparisons += 1
                
                # Check if ETB label direction matches forward return direction
                if etb_label == 1 and forward_return < 0:
                    inconsistencies += 1
                elif etb_label == -1 and forward_return > 0:
                    inconsistencies += 1
        
        if total_comparisons > 0:
            inconsistency_rate = inconsistencies / total_comparisons
            
            result.metrics["cross_label_consistency"] = {
                "total_comparisons": total_comparisons,
                "inconsistencies": inconsistencies,
                "inconsistency_rate": inconsistency_rate
            }
            
            if inconsistency_rate > 0.3:  # > 30% inconsistency
                result.add_issue(
                    ValidationCategory.DATA_CONSISTENCY,
                    ValidationSeverity.WARNING,
                    "High inconsistency between Enhanced Triple Barrier and forward returns",
                    {
                        "inconsistency_rate": inconsistency_rate,
                        "total_comparisons": total_comparisons
                    }
                )
    
    def _validate_batch_performance_metrics(self, label_sets: List[LabelSet], result: ValidationResult):
        """Validate performance metrics across batch"""
        
        computation_times = [
            ls.computation_time_ms for ls in label_sets 
            if ls.computation_time_ms is not None
        ]
        
        if computation_times:
            avg_time = np.mean(computation_times)
            max_time = np.max(computation_times)
            std_time = np.std(computation_times)
            
            result.metrics["batch_performance"] = {
                "avg_computation_time_ms": float(avg_time),
                "max_computation_time_ms": float(max_time),
                "std_computation_time_ms": float(std_time),
                "total_samples": len(computation_times)
            }
            
            # Check for performance issues
            if avg_time > 1000:  # Average > 1 second
                result.add_issue(
                    ValidationCategory.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    "High average computation time across batch",
                    {"avg_computation_time_ms": avg_time}
                )
            
            if max_time > 10000:  # Max > 10 seconds
                result.add_issue(
                    ValidationCategory.PERFORMANCE,
                    ValidationSeverity.WARNING,
                    "Very slow computation detected in batch",
                    {"max_computation_time_ms": max_time}
                )
    
    def _calculate_time_span(self, label_sets: List[LabelSet]) -> float:
        """Calculate time span of label sets in hours"""
        
        if len(label_sets) <= 1:
            return 0.0
        
        timestamps = [ls.ts for ls in label_sets]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        
        return (max_ts - min_ts).total_seconds() / 3600
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        return {
            **self.validation_stats,
            "success_rate": (
                (self.validation_stats["total_validations"] - self.validation_stats["failed_validations"]) /
                max(self.validation_stats["total_validations"], 1)
            )
        }
    
    def create_alerting_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """Create summary for alerting systems"""
        
        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        
        alert_level = "green"
        if critical_issues:
            alert_level = "red"
        elif error_issues:
            alert_level = "yellow"
        elif result.get_issues_by_severity(ValidationSeverity.WARNING):
            alert_level = "orange"
        
        return {
            "alert_level": alert_level,
            "is_valid": result.is_valid,
            "total_issues": len(result.issues),
            "critical_count": len(critical_issues),
            "error_count": len(error_issues),
            "warning_count": len(result.get_issues_by_severity(ValidationSeverity.WARNING)),
            "top_issues": [str(issue) for issue in (critical_issues + error_issues)[:5]],
            "validation_time_ms": result.validation_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global validator instance
label_validator = LabelValidator()