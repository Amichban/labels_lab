/**
 * TypeScript types for Label Computation System API
 * Generated from OpenAPI 3.0 specification
 * 
 * These types provide compile-time validation and IntelliSense support
 * for all API request and response schemas.
 */

// ===========================================================================
// ENUMS
// ===========================================================================

export enum Granularity {
  M15 = 'M15',
  H1 = 'H1',
  H4 = 'H4',
  D = 'D',
  W = 'W'
}

export enum JobStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  PAUSED = 'paused'
}

export enum HealthStatus {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  UNHEALTHY = 'unhealthy'
}

export enum BarrierHit {
  UPPER = 'upper',
  LOWER = 'lower',
  NONE = 'none'
}

export enum CacheStrategy {
  PREFER_CACHE = 'prefer_cache',
  BYPASS_CACHE = 'bypass_cache',
  REFRESH_CACHE = 'refresh_cache'
}

export enum Priority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high'
}

// ===========================================================================
// BASE TYPES
// ===========================================================================

export type InstrumentId = string; // Pattern: /^[A-Z]{6}$|^[A-Z0-9]+$/
export type JobId = string; // Pattern: /^bf_[0-9]{8}_[a-z0-9]+_[a-z0-9]+_[a-z0-9]{6}$/
export type Timestamp = string; // ISO 8601 datetime
export type TripleBarrierLabel = -1 | 0 | 1;
export type ReturnQuantile = number; // 0-100
export type Percentage = number; // 0-100
export type Probability = number; // 0-1

// ===========================================================================
// REQUEST TYPES
// ===========================================================================

export interface CandleData {
  /** Candle timestamp (aligned to granularity boundary) */
  ts: Timestamp;
  /** Opening price */
  open: number;
  /** Highest price */
  high: number;
  /** Lowest price */
  low: number;
  /** Closing price */
  close: number;
  /** Volume */
  volume: number;
  /** 14-period Average True Range (required for some labels) */
  atr_14?: number;
}

export interface LabelComputeOptions {
  /** Forward-looking horizon in periods (1-100) */
  horizon_periods?: number;
  /** Whether to use cached results */
  use_cache?: boolean;
  /** Force recomputation even if cached */
  force_recompute?: boolean;
}

export interface CandleLabelRequest {
  /** Instrument identifier (e.g., EURUSD, GBPJPY) */
  instrument_id: InstrumentId;
  /** Time granularity for labels */
  granularity: Granularity;
  /** Candle data for label computation */
  candle: CandleData;
  /** Specific label types to compute (optional, computes all if empty) */
  label_types?: string[];
  /** Computation options */
  options?: LabelComputeOptions;
}

export interface BatchBackfillOptions {
  /** Candles to process per chunk (1000-50000) */
  chunk_size?: number;
  /** Number of parallel workers (1-16) */
  parallel_workers?: number;
  /** Recompute existing labels */
  force_recompute?: boolean;
  /** Job priority */
  priority?: Priority;
}

export interface BatchBackfillRequest {
  /** Instrument identifier */
  instrument_id: InstrumentId;
  /** Time granularity */
  granularity: Granularity;
  /** Start date (inclusive) */
  start_date: Timestamp;
  /** End date (exclusive) */
  end_date: Timestamp;
  /** Label types to compute (all if empty) */
  label_types?: string[];
  /** Backfill options */
  options?: BatchBackfillOptions;
}

export interface CacheWarmRequest {
  /** Instrument to warm cache for */
  instrument_id: InstrumentId;
  /** Granularity to warm cache for */
  granularity: Granularity;
  /** Hours of data to cache (1-168) */
  hours?: number;
}

// ===========================================================================
// RESPONSE TYPES
// ===========================================================================

export interface EnhancedTripleBarrierLabel {
  /** -1=lower barrier hit, 0=no barrier hit, 1=upper barrier hit */
  label: TripleBarrierLabel;
  /** Which barrier was hit first */
  barrier_hit: BarrierHit;
  /** Periods until barrier hit (or horizon if none) */
  time_to_barrier: number;
  /** Price of barrier that was hit */
  barrier_price?: number | null;
  /** Whether barriers were adjusted based on S/R levels */
  level_adjusted: boolean;
  /** Nearest support level price */
  nearest_support?: number | null;
  /** Nearest resistance level price */
  nearest_resistance?: number | null;
}

export interface VolScaledReturnLabel {
  /** Volatility-scaled return value */
  value: number;
  /** Quantile of the scaled return (0-1) */
  quantile: Probability;
  /** Raw forward return before scaling */
  raw_return?: number;
  /** ATR-based volatility scaling factor */
  volatility_factor?: number;
}

export interface MfeMaeLabel {
  /** Maximum Favorable Excursion */
  mfe: number;
  /** Maximum Adverse Excursion */
  mae: number;
  /** Ratio of MFE to MAE */
  profit_factor: number;
  /** Periods to reach MFE */
  mfe_time?: number;
  /** Periods to reach MAE */
  mae_time?: number;
}

export interface LabelValues {
  enhanced_triple_barrier?: EnhancedTripleBarrierLabel;
  vol_scaled_return?: VolScaledReturnLabel;
  mfe_mae?: MfeMaeLabel;
  /** Return quantile bucket (0-100) */
  return_quantile?: ReturnQuantile;
  /** Forward return over horizon */
  forward_return?: number;
  /** Allow additional label types */
  [key: string]: any;
}

export interface ComputedLabels {
  instrument_id: InstrumentId;
  granularity: Granularity;
  ts: Timestamp;
  /** Computed label values */
  labels: LabelValues;
  /** Time taken to compute labels */
  computation_time_ms: number;
  /** Whether result came from cache */
  cache_hit: boolean;
  /** Label computation version */
  version: string;
}

export interface HateoasLinks {
  self?: string;
  status?: string;
  cancel?: string;
}

export interface BatchJobResponse {
  /** Unique job identifier */
  job_id: JobId;
  /** Initial job status */
  status: string;
  /** Estimated completion time in minutes */
  estimated_duration_minutes: number;
  /** Total candles to process */
  estimated_candles: number;
  /** Job priority */
  priority: Priority;
  /** HATEOAS links for job operations */
  _links?: HateoasLinks;
}

export interface JobProgress {
  completed_candles: number;
  total_candles: number;
  percentage: Percentage;
  /** Current processing date */
  current_date?: Timestamp;
  chunks_completed?: number;
  chunks_total?: number;
}

export interface JobPerformance {
  candles_per_minute?: number;
  avg_compute_time_ms?: number;
  cache_hit_rate?: Probability;
  error_rate?: Probability;
}

export interface BatchJobStatus {
  job_id: JobId;
  status: JobStatus;
  /** Job progress information */
  progress: JobProgress;
  /** Performance metrics */
  performance?: JobPerformance;
  estimated_completion?: Timestamp | null;
  created_at: Timestamp;
  updated_at: Timestamp;
  /** Error message if status is failed */
  error_message?: string | null;
}

export interface PaginationInfo {
  /** Current page number */
  page: number;
  /** Items per page */
  per_page: number;
  /** Total items count */
  total: number;
  /** Total pages count */
  total_pages: number;
  /** Whether next page exists */
  has_next: boolean;
  /** Whether previous page exists */
  has_prev: boolean;
  /** Next page number */
  next_page?: number | null;
  /** Previous page number */
  prev_page?: number | null;
}

export interface BatchJobsList {
  data: BatchJobStatus[];
  pagination: PaginationInfo;
}

export interface LabelsList {
  data: ComputedLabels[];
  pagination: PaginationInfo;
}

// ===========================================================================
// HEALTH & MONITORING TYPES
// ===========================================================================

export interface HealthMetrics {
  cache_hit_rate?: Probability;
  avg_computation_ms?: number;
  active_batch_jobs?: number;
  labels_computed_last_hour?: number;
}

export interface HealthResponse {
  status: HealthStatus;
  /** Application version */
  version: string;
  timestamp: Timestamp;
  uptime_seconds?: number;
  /** Health check results for dependencies */
  checks?: Record<string, 'ok' | 'warning' | 'error'>;
  metrics?: HealthMetrics;
  /** Error messages for failed checks */
  errors?: string[];
}

export interface PerformanceMetrics {
  avg_computation_time_ms?: number;
  p50_computation_time_ms?: number;
  p95_computation_time_ms?: number;
  p99_computation_time_ms?: number;
  requests_per_second?: number;
  error_rate?: Probability;
}

export interface CacheMetrics {
  hit_rate?: Probability;
  memory_usage_mb?: number;
  evictions_per_minute?: number;
  keys_total?: number;
}

export interface BusinessMetrics {
  labels_computed_total?: number;
  unique_instruments?: number;
  active_batch_jobs?: number;
  avg_batch_throughput_candles_per_min?: number;
}

export interface MetricsResponse {
  timestamp: Timestamp;
  /** Time window for metrics */
  window: '5m' | '15m' | '1h' | '6h' | '24h';
  performance?: PerformanceMetrics;
  cache?: CacheMetrics;
  business?: BusinessMetrics;
}

export interface CacheLevelStats {
  keys_count: number;
  hit_rate: Probability;
  avg_ttl_minutes: number;
}

export interface RedisStats {
  memory_usage_mb: number;
  keys_total: number;
  hit_rate: Probability;
  evictions_last_hour: number;
  connections_active: number;
}

export interface CacheStatsResponse {
  timestamp: Timestamp;
  redis: RedisStats;
  cache_levels: {
    labels: CacheLevelStats;
    lookback_data: CacheLevelStats;
    levels: CacheLevelStats;
  };
}

// ===========================================================================
// ERROR TYPES
// ===========================================================================

export interface ErrorDetail {
  /** Field name with error */
  field?: string;
  /** Field-specific error message */
  message: string;
  /** Field-specific error code */
  code?: string;
}

export interface ErrorInfo {
  /** Machine-readable error code */
  code: string;
  /** Human-readable error message */
  message: string;
  /** Detailed validation errors */
  details?: ErrorDetail[];
  /** Request trace ID for debugging */
  trace_id?: string;
}

export interface ErrorResponse {
  error: ErrorInfo;
}

// ===========================================================================
// QUERY PARAMETERS
// ===========================================================================

export interface LabelsQueryParams {
  instrument_id: InstrumentId;
  granularity: Granularity;
  start_date: Timestamp;
  end_date: Timestamp;
  label_types?: string;
  enhanced_triple_barrier_label?: TripleBarrierLabel;
  return_quantile_min?: ReturnQuantile;
  return_quantile_max?: ReturnQuantile;
  page?: number;
  per_page?: number;
  sort?: 'ts_asc' | 'ts_desc' | 'forward_return_asc' | 'forward_return_desc';
}

export interface BatchJobsQueryParams {
  status?: JobStatus;
  instrument_id?: InstrumentId;
  granularity?: Granularity;
  page?: number;
  per_page?: number;
  sort?: 'created_at_desc' | 'created_at_asc' | 'updated_at_desc';
}

export interface MetricsQueryParams {
  format?: 'json' | 'prometheus';
  window?: '5m' | '15m' | '1h' | '6h' | '24h';
}

export interface CacheInvalidateParams {
  instrument_id?: InstrumentId;
  granularity?: Granularity;
  pattern?: string;
}

// ===========================================================================
// HTTP HEADERS
// ===========================================================================

export interface ApiHeaders {
  'Content-Type'?: 'application/json';
  'Authorization'?: string; // Bearer token
  'X-API-Key'?: string; // API key
  'X-Request-ID'?: string; // UUID for tracing
  'X-Cache-Strategy'?: CacheStrategy;
  'Accept'?: 'application/json' | 'text/plain'; // For metrics endpoint
}

export interface ResponseHeaders {
  'X-Compute-Time-Ms'?: string;
  'X-Cache-Hit'?: string;
  'X-Rate-Limit-Remaining'?: string;
  'X-Query-Time-Ms'?: string;
  'X-RateLimit-Limit'?: string;
  'X-RateLimit-Remaining'?: string;
  'X-RateLimit-Reset'?: string;
  'Retry-After'?: string;
  'WWW-Authenticate'?: string;
}

// ===========================================================================
// API CLIENT TYPES
// ===========================================================================

export interface ApiClientConfig {
  baseUrl: string;
  apiKey?: string;
  bearerToken?: string;
  timeout?: number;
  retryAttempts?: number;
  defaultHeaders?: Record<string, string>;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, string>;
}

export interface ApiError extends Error {
  status?: number;
  code?: string;
  details?: ErrorDetail[];
  trace_id?: string;
}

// ===========================================================================
// UTILITY TYPES
// ===========================================================================

/** Extract the data type from a paginated response */
export type ExtractListData<T extends { data: any[] }> = T['data'][0];

/** Make all properties optional for partial updates */
export type PartialUpdate<T> = Partial<T>;

/** Pick specific fields for projections */
export type ProjectedFields<T, K extends keyof T> = Pick<T, K>;

/** Union of all supported label type names */
export type LabelType = 
  | 'enhanced_triple_barrier'
  | 'vol_scaled_return'
  | 'mfe_mae'
  | 'return_quantile'
  | 'forward_return';

/** Union of all supported error codes */
export type ErrorCode =
  | 'BAD_REQUEST'
  | 'UNAUTHORIZED'
  | 'VALIDATION_ERROR'
  | 'NOT_FOUND'
  | 'RATE_LIMIT_EXCEEDED'
  | 'INTERNAL_ERROR'
  | 'BACKFILL_IN_PROGRESS';

/** Union of all HTTP status codes used by the API */
export type HttpStatusCode = 200 | 201 | 202 | 400 | 401 | 404 | 409 | 422 | 429 | 500 | 503;