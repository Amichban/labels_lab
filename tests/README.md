# Enhanced Triple Barrier Label 11.a - Test Suite

This comprehensive test suite validates the Label 11.a Enhanced Triple Barrier implementation, focusing on critical aspects like timestamp alignment, barrier computation, multi-timeframe coordination, and performance.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_timestamp_aligner.py     # TimestampAligner tests
│   ├── test_label_computation.py     # LabelComputationEngine tests
│   └── test_edge_cases.py            # Edge case and boundary tests
├── integration/                # Integration tests (services working together)
│   └── test_label_pipeline.py       # End-to-end pipeline tests
├── performance/                # Performance and benchmark tests
│   └── test_benchmarks.py           # Throughput, latency, memory tests
└── README.md                   # This file
```

## Key Test Areas

### 1. Timestamp Alignment (CRITICAL)
- **H4 alignment at 1,5,9,13,17,21 UTC** - Essential for preventing look-ahead bias
- Multi-timeframe alignment consistency
- Edge cases: DST transitions, leap years, boundaries
- Performance: 10k+ alignment operations per second

### 2. Enhanced Triple Barrier Computation
- ATR-based barrier sizing
- S/R level adjustments with buffer zones
- Path granularity usage (H4→H1, D→H4, etc.)
- Barrier hit detection accuracy
- No look-ahead bias validation

### 3. Integration Pipeline
- End-to-end label computation flow
- ClickHouse and Redis service integration
- Multi-timeframe data coordination
- Error handling and recovery
- Cache hit/miss scenarios

### 4. Performance Benchmarks
- Cache performance: <1ms for hits, <50ms for misses
- Batch processing: >10 candles/sec
- Memory usage: <100KB per computation
- Latency distribution: P95 <50ms, P99 <100ms

## Running Tests

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (recommended)
python scripts/run_tests.py --pyramid --coverage

# Run specific test types
python scripts/run_tests.py --unit
python scripts/run_tests.py --integration  
python scripts/run_tests.py --performance
```

### Testing Pyramid Strategy
The pyramid strategy runs tests in optimal order:

1. **Unit Tests** - Fast feedback, fail fast
2. **Integration Tests** - Component interactions  
3. **Performance Tests** - Benchmarks and stress tests

```bash
# Use pyramid strategy (recommended)
python scripts/run_tests.py --pyramid

# Skip slow tests for rapid feedback
python scripts/run_tests.py --pyramid --fast
```

### Specific Test Categories

```bash
# Critical H4 alignment tests
pytest tests/unit/test_timestamp_aligner.py::TestTimestampAligner::test_h4_alignment_exact_boundaries -v

# Enhanced Triple Barrier core logic
pytest tests/unit/test_label_computation.py::TestLabelComputationEngine::test_compute_enhanced_triple_barrier_upper_hit -v

# End-to-end pipeline validation
pytest tests/integration/test_label_pipeline.py::TestLabelComputationPipeline::test_complete_pipeline_h4_candle -v

# Performance benchmarks
pytest tests/performance/test_benchmarks.py::TestCachePerformanceBenchmarks -v
```

### Coverage and Quality

```bash
# Generate coverage report
python scripts/run_tests.py --unit --coverage

# View coverage report
open htmlcov/index.html

# Run with specific markers
pytest -m "unit and timestamp_aligner" -v
pytest -m "edge_cases and not slow" -v
```

## Test Markers

Tests are organized with markers for flexible execution:

- `unit` - Unit tests for individual components
- `integration` - Integration tests for component interactions
- `performance` - Performance and benchmark tests  
- `edge_cases` - Edge cases and boundary conditions
- `slow` - Tests taking >5 seconds

Component-specific markers:
- `timestamp_aligner` - TimestampAligner tests
- `label_computation` - LabelComputationEngine tests
- `cache_tests` - Caching functionality tests

## Critical Test Cases

### H4 Alignment Validation
```python
def test_h4_alignment_exact_boundaries():
    """CRITICAL: Validates H4 candles align to 1,5,9,13,17,21 UTC"""
    # This test prevents look-ahead bias in Label 11.a
```

### Look-Ahead Bias Prevention
```python
def test_no_lookahead_bias_path_data():
    """CRITICAL: Ensures no future data is used in computation"""
    # Validates all path data timestamps >= candle timestamp
```

### Barrier Adjustment Logic
```python  
def test_compute_enhanced_triple_barrier_with_sr_adjustments():
    """CRITICAL: Tests S/R level barrier adjustments with buffers"""
    # Validates correct level-based barrier placement
```

## Performance Requirements

The test suite enforces these performance requirements:

| Metric | Requirement | Test Location |
|--------|-------------|---------------|
| H4 Alignment | >10k ops/sec | test_timestamp_aligner.py |
| Cache Hits | <1ms average | test_benchmarks.py |
| Cache Misses | <50ms average | test_benchmarks.py |  
| Batch Processing | >10 candles/sec | test_benchmarks.py |
| Memory per Computation | <100KB | test_benchmarks.py |
| P95 Latency | <50ms | test_benchmarks.py |
| P99 Latency | <100ms | test_benchmarks.py |

## Test Data and Fixtures

### Realistic Test Data
- **H4-aligned timestamps** - Proper 1,5,9,13,17,21 UTC alignment
- **Realistic S/R levels** - Based on actual market behavior
- **Path data scenarios** - Upper hit, lower hit, no hit cases
- **Edge case timestamps** - DST, leap years, boundaries

### Mock Services  
- **ClickHouse Service** - Controlled data responses
- **Redis Cache** - Hit/miss scenario simulation
- **Performance Timers** - Accurate latency measurement

## Debugging Failed Tests

### Common Issues

1. **H4 Alignment Failures**
   ```bash
   # Check specific alignment case
   pytest tests/unit/test_timestamp_aligner.py::test_h4_alignment_between_boundaries -v -s
   ```

2. **Barrier Computation Issues**
   ```bash
   # Debug barrier logic with verbose output
   pytest tests/unit/test_label_computation.py::test_compute_enhanced_triple_barrier_upper_hit -v -s --tb=long
   ```

3. **Performance Regressions**
   ```bash  
   # Run performance tests with detailed output
   pytest tests/performance/test_benchmarks.py -v -s
   ```

### Test Environment Setup

```bash
# Ensure clean test environment
pip install -e .
python -m pytest --collect-only  # Verify test discovery

# Check for import issues
python -c "from src.core.label_computation import LabelComputationEngine; print('OK')"
```

## Continuous Integration

For CI/CD integration:

```yaml
# .github/workflows/tests.yml
- name: Run Test Suite
  run: |
    python scripts/run_tests.py --pyramid --coverage --junit-xml=test-results.xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v2
  with:
    file: coverage.xml
```

## Contributing Test Guidelines

When adding tests:

1. **Follow pyramid structure** - More unit tests, fewer integration tests
2. **Use descriptive names** - `test_h4_alignment_weekend_boundary` not `test_alignment`
3. **Include edge cases** - DST transitions, leap years, precision issues  
4. **Add performance constraints** - Assert timing and memory requirements
5. **Mock external dependencies** - Use fixtures for services
6. **Document critical tests** - Explain why a test is important

### Test Naming Convention

```python
def test_[component]_[scenario]_[expected_outcome]():
    """Brief description of what this test validates and why it's important."""
    # Arrange
    # Act  
    # Assert
```

Example:
```python
def test_h4_alignment_early_morning_aligns_to_previous_day():
    """Tests that times before 1:00 UTC align to previous day's 21:00.
    
    CRITICAL: This prevents look-ahead bias when H4 candles span midnight.
    """
```

## Performance Monitoring

The test suite includes built-in performance monitoring:

- **Execution time tracking** for each test category
- **Memory usage measurement** during test runs
- **Throughput validation** for batch operations
- **Latency distribution analysis** for real-time scenarios

Monitor performance trends over time to catch regressions early.

---

For questions about the test suite, refer to the test-runner guidelines at `.claude/subagents/test-runner.md` or check the inline test documentation.