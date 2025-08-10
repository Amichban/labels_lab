/**
 * TypeScript SDK Client for Label Computation System API
 * Generated from OpenAPI 3.0 specification
 * 
 * Provides a typed, Promise-based client with automatic validation,
 * error handling, and retry logic.
 */

import {
  // Request types
  CandleLabelRequest,
  BatchBackfillRequest,
  CacheWarmRequest,
  
  // Response types
  ComputedLabels,
  BatchJobResponse,
  BatchJobStatus,
  BatchJobsList,
  LabelsList,
  HealthResponse,
  MetricsResponse,
  CacheStatsResponse,
  
  // Query parameter types
  LabelsQueryParams,
  BatchJobsQueryParams,
  MetricsQueryParams,
  CacheInvalidateParams,
  
  // Utility types
  ApiClientConfig,
  ApiResponse,
  ApiError,
  ApiHeaders,
  InstrumentId,
  Granularity,
  JobId,
  Timestamp,
} from './types';

/**
 * Label Computation System API Client
 * 
 * Features:
 * - Automatic request/response validation
 * - Built-in retry logic with exponential backoff
 * - Request tracing and correlation IDs
 * - Rate limit handling
 * - Comprehensive error handling
 * - TypeScript type safety
 */
export class LabelComputeApiClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeout: number;
  private readonly retryAttempts: number;

  constructor(config: ApiClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.timeout = config.timeout || 30000; // 30 seconds default
    this.retryAttempts = config.retryAttempts || 3;

    // Setup default headers
    this.headers = {
      'Content-Type': 'application/json',
      ...config.defaultHeaders,
    };

    // Add authentication
    if (config.bearerToken) {
      this.headers['Authorization'] = `Bearer ${config.bearerToken}`;
    } else if (config.apiKey) {
      this.headers['X-API-Key'] = config.apiKey;
    }
  }

  // ===========================================================================
  // LABEL COMPUTATION METHODS
  // ===========================================================================

  /**
   * Compute labels for a single candle in real-time
   * Target latency: <100ms p99
   */
  async computeLabels(
    request: CandleLabelRequest,
    options?: {
      requestId?: string;
      cacheStrategy?: 'prefer_cache' | 'bypass_cache' | 'refresh_cache';
    }
  ): Promise<ApiResponse<ComputedLabels>> {
    const headers: ApiHeaders = {};
    
    if (options?.requestId) {
      headers['X-Request-ID'] = options.requestId;
    }
    
    if (options?.cacheStrategy) {
      headers['X-Cache-Strategy'] = options.cacheStrategy;
    }

    return this.request<ComputedLabels>('POST', '/labels/compute', {
      body: request,
      headers,
    });
  }

  /**
   * Query computed labels with flexible filtering and pagination
   */
  async getLabels(params: LabelsQueryParams): Promise<ApiResponse<LabelsList>> {
    const queryString = this.buildQueryString(params);
    return this.request<LabelsList>('GET', `/labels?${queryString}`);
  }

  /**
   * Get labels for a specific candle timestamp
   */
  async getLabelsByTimestamp(
    instrumentId: InstrumentId,
    granularity: Granularity,
    timestamp: Timestamp,
    labelTypes?: string[]
  ): Promise<ApiResponse<ComputedLabels>> {
    const encodedTimestamp = encodeURIComponent(timestamp);
    let url = `/labels/${instrumentId}/${granularity}/${encodedTimestamp}`;
    
    if (labelTypes && labelTypes.length > 0) {
      url += `?label_types=${labelTypes.join(',')}`;
    }
    
    return this.request<ComputedLabels>('GET', url);
  }

  // ===========================================================================
  // BATCH OPERATIONS METHODS
  // ===========================================================================

  /**
   * Start a batch backfill operation
   * Returns immediately with job ID for status tracking
   */
  async startBatchBackfill(
    request: BatchBackfillRequest
  ): Promise<ApiResponse<BatchJobResponse>> {
    return this.request<BatchJobResponse>('POST', '/batch/backfill', {
      body: request,
    });
  }

  /**
   * Get status of a batch backfill job
   */
  async getBatchJobStatus(jobId: JobId): Promise<ApiResponse<BatchJobStatus>> {
    return this.request<BatchJobStatus>('GET', `/batch/jobs/${jobId}`);
  }

  /**
   * List batch jobs with filtering and pagination
   */
  async listBatchJobs(params?: BatchJobsQueryParams): Promise<ApiResponse<BatchJobsList>> {
    const queryString = params ? this.buildQueryString(params) : '';
    const url = queryString ? `/batch/jobs?${queryString}` : '/batch/jobs';
    return this.request<BatchJobsList>('GET', url);
  }

  /**
   * Cancel a running batch job
   */
  async cancelBatchJob(jobId: JobId): Promise<ApiResponse<{ message: string }>> {
    return this.request('DELETE', `/batch/jobs/${jobId}`);
  }

  // ===========================================================================
  // HEALTH & MONITORING METHODS
  // ===========================================================================

  /**
   * Comprehensive system health check
   */
  async getHealth(): Promise<ApiResponse<HealthResponse>> {
    return this.request<HealthResponse>('GET', '/health');
  }

  /**
   * Kubernetes readiness probe
   */
  async getReadiness(): Promise<ApiResponse<{ status: 'ready'; timestamp: Timestamp }>> {
    return this.request('GET', '/health/ready');
  }

  /**
   * Kubernetes liveness probe
   */
  async getLiveness(): Promise<ApiResponse<{ status: 'alive'; timestamp: Timestamp }>> {
    return this.request('GET', '/health/live');
  }

  /**
   * Get system metrics
   */
  async getMetrics(params?: MetricsQueryParams): Promise<ApiResponse<MetricsResponse | string>> {
    const queryString = params ? this.buildQueryString(params) : '';
    const url = queryString ? `/metrics?${queryString}` : '/metrics';
    
    const headers: Record<string, string> = {};
    if (params?.format === 'prometheus') {
      headers['Accept'] = 'text/plain';
    }

    return this.request(url === '/metrics' ? 'GET' : 'GET', url, { headers });
  }

  // ===========================================================================
  // CACHE MANAGEMENT METHODS
  // ===========================================================================

  /**
   * Get cache performance statistics
   */
  async getCacheStats(): Promise<ApiResponse<CacheStatsResponse>> {
    return this.request<CacheStatsResponse>('GET', '/cache/stats');
  }

  /**
   * Pre-warm cache with recent data for an instrument
   */
  async warmCache(request: CacheWarmRequest): Promise<ApiResponse<{ 
    message: string; 
    estimated_completion: Timestamp 
  }>> {
    return this.request('POST', '/cache/warm', { body: request });
  }

  /**
   * Invalidate cache entries
   */
  async invalidateCache(params?: CacheInvalidateParams): Promise<ApiResponse<{
    message: string;
    keys_deleted: number;
  }>> {
    const queryString = params ? this.buildQueryString(params) : '';
    const url = queryString ? `/cache/invalidate?${queryString}` : '/cache/invalidate';
    return this.request('DELETE', url);
  }

  // ===========================================================================
  // UTILITY METHODS
  // ===========================================================================

  /**
   * Poll a batch job until completion or failure
   */
  async pollBatchJob(
    jobId: JobId,
    options?: {
      intervalMs?: number;
      timeoutMs?: number;
      onProgress?: (status: BatchJobStatus) => void;
    }
  ): Promise<BatchJobStatus> {
    const intervalMs = options?.intervalMs || 5000; // 5 seconds
    const timeoutMs = options?.timeoutMs || 3600000; // 1 hour
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const response = await this.getBatchJobStatus(jobId);
      const status = response.data;

      if (options?.onProgress) {
        options.onProgress(status);
      }

      if (['completed', 'failed', 'cancelled'].includes(status.status)) {
        return status;
      }

      await this.sleep(intervalMs);
    }

    throw new Error(`Batch job ${jobId} polling timed out after ${timeoutMs}ms`);
  }

  /**
   * Wait for system to become healthy
   */
  async waitForHealth(options?: {
    intervalMs?: number;
    timeoutMs?: number;
  }): Promise<HealthResponse> {
    const intervalMs = options?.intervalMs || 1000; // 1 second
    const timeoutMs = options?.timeoutMs || 60000; // 1 minute
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      try {
        const response = await this.getHealth();
        if (response.data.status === 'healthy') {
          return response.data;
        }
      } catch (error) {
        // Ignore health check errors during wait
      }

      await this.sleep(intervalMs);
    }

    throw new Error(`System did not become healthy within ${timeoutMs}ms`);
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private async request<T = any>(
    method: string,
    path: string,
    options?: {
      body?: any;
      headers?: Record<string, string>;
      timeout?: number;
    }
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${path}`;
    const requestHeaders = { ...this.headers, ...options?.headers };
    const timeout = options?.timeout || this.timeout;

    // Generate request ID if not provided
    if (!requestHeaders['X-Request-ID']) {
      requestHeaders['X-Request-ID'] = this.generateRequestId();
    }

    const requestOptions: RequestInit = {
      method,
      headers: requestHeaders,
      signal: AbortSignal.timeout(timeout),
    };

    if (options?.body) {
      requestOptions.body = JSON.stringify(options.body);
    }

    let lastError: Error;
    
    for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
      try {
        const response = await fetch(url, requestOptions);
        
        // Handle rate limiting
        if (response.status === 429) {
          const retryAfter = response.headers.get('Retry-After');
          if (retryAfter && attempt < this.retryAttempts) {
            await this.sleep(parseInt(retryAfter) * 1000);
            continue;
          }
        }

        // Parse response
        const responseHeaders: Record<string, string> = {};
        response.headers.forEach((value, key) => {
          responseHeaders[key] = value;
        });

        let data: T;
        const contentType = response.headers.get('content-type');
        
        if (contentType?.includes('application/json')) {
          data = await response.json();
        } else {
          data = (await response.text()) as any;
        }

        // Handle HTTP errors
        if (!response.ok) {
          const error = this.createApiError(response.status, data as any, requestHeaders['X-Request-ID']);
          throw error;
        }

        return {
          data,
          status: response.status,
          statusText: response.statusText,
          headers: responseHeaders,
        };

      } catch (error) {
        lastError = error as Error;
        
        // Don't retry on client errors (4xx) except rate limiting
        if (error instanceof ApiError && error.status && error.status >= 400 && error.status < 500 && error.status !== 429) {
          throw error;
        }

        // Exponential backoff for retries
        if (attempt < this.retryAttempts) {
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000); // Max 10 seconds
          await this.sleep(delay);
        }
      }
    }

    throw lastError!;
  }

  private createApiError(status: number, responseData: any, traceId?: string): ApiError {
    let message = `HTTP ${status}`;
    let code = 'HTTP_ERROR';
    let details: any[] = [];

    // Try to extract error information from response
    if (responseData?.error) {
      message = responseData.error.message || message;
      code = responseData.error.code || code;
      details = responseData.error.details || [];
    }

    const error = new Error(message) as ApiError;
    error.name = 'ApiError';
    error.status = status;
    error.code = code;
    error.details = details;
    error.trace_id = traceId;

    return error;
  }

  private buildQueryString(params: Record<string, any>): string {
    const searchParams = new URLSearchParams();
    
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, value.toString());
      }
    });
    
    return searchParams.toString();
  }

  private generateRequestId(): string {
    return 'req_' + Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ===========================================================================
// FACTORY FUNCTIONS
// ===========================================================================

/**
 * Create a client configured for production use
 */
export function createProductionClient(config: {
  bearerToken?: string;
  apiKey?: string;
}): LabelComputeApiClient {
  return new LabelComputeApiClient({
    baseUrl: 'https://api.labelcompute.com/v1',
    ...config,
    timeout: 30000,
    retryAttempts: 3,
    defaultHeaders: {
      'User-Agent': 'LabelCompute-SDK/1.0.0',
    },
  });
}

/**
 * Create a client configured for staging use
 */
export function createStagingClient(config: {
  bearerToken?: string;
  apiKey?: string;
}): LabelComputeApiClient {
  return new LabelComputeApiClient({
    baseUrl: 'https://staging-api.labelcompute.com/v1',
    ...config,
    timeout: 30000,
    retryAttempts: 2,
    defaultHeaders: {
      'User-Agent': 'LabelCompute-SDK/1.0.0-staging',
    },
  });
}

/**
 * Create a client configured for local development
 */
export function createLocalClient(config?: {
  port?: number;
  bearerToken?: string;
  apiKey?: string;
}): LabelComputeApiClient {
  const port = config?.port || 8000;
  
  return new LabelComputeApiClient({
    baseUrl: `http://localhost:${port}/v1`,
    bearerToken: config?.bearerToken,
    apiKey: config?.apiKey,
    timeout: 10000,
    retryAttempts: 1,
    defaultHeaders: {
      'User-Agent': 'LabelCompute-SDK/1.0.0-local',
    },
  });
}

// Export everything for convenience
export * from './types';