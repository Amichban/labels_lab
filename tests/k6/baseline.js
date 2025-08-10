import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const readLatency = new Trend('read_latency');
const writeLatency = new Trend('write_latency');
const searchLatency = new Trend('search_latency');

export let options = {
  stages: [
    { duration: '30s', target: 50 },   // Ramp up to 50 users
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '3m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 200 },   // Spike to 200 users
    { duration: '3m', target: 200 },   // Stay at 200 users
    { duration: '2m', target: 0 },     // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],  // 95% under 500ms, 99% under 1s
    errors: ['rate<0.01'],                            // Error rate under 1%
    read_latency: ['p(95)<200'],                      // Read p95 under 200ms
    write_latency: ['p(95)<500'],                     // Write p95 under 500ms
    search_latency: ['p(95)<1000'],                   // Search p95 under 1s
  },
};

const BASE_URL = __ENV.API_URL || 'http://localhost:8000';

export default function() {
  // Simulate realistic user behavior
  const userId = Math.floor(Math.random() * 10000) + 1;
  
  // 1. Read heavy operation (80% of traffic)
  if (Math.random() < 0.8) {
    // Get user profile
    let startTime = Date.now();
    let res = http.get(`${BASE_URL}/api/users/${userId}`);
    readLatency.add(Date.now() - startTime);
    
    check(res, {
      'read status is 200': (r) => r.status === 200,
      'read response time < 200ms': (r) => r.timings.duration < 200,
    }) || errorRate.add(1);
    
    // Get user's posts (pagination)
    startTime = Date.now();
    res = http.get(`${BASE_URL}/api/posts?user_id=${userId}&limit=20&offset=0`);
    readLatency.add(Date.now() - startTime);
    
    check(res, {
      'posts status is 200': (r) => r.status === 200,
      'posts has data': (r) => JSON.parse(r.body).data !== undefined,
    }) || errorRate.add(1);
  }
  
  // 2. Write operation (15% of traffic)
  if (Math.random() < 0.15) {
    const payload = JSON.stringify({
      title: `Test Post ${Date.now()}`,
      content: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.',
      user_id: userId,
      status: 'published',
    });
    
    const params = {
      headers: { 'Content-Type': 'application/json' },
    };
    
    let startTime = Date.now();
    let res = http.post(`${BASE_URL}/api/posts`, payload, params);
    writeLatency.add(Date.now() - startTime);
    
    check(res, {
      'write status is 201': (r) => r.status === 201,
      'write response time < 500ms': (r) => r.timings.duration < 500,
    }) || errorRate.add(1);
  }
  
  // 3. Search operation (5% of traffic)
  if (Math.random() < 0.05) {
    const searchTerm = ['javascript', 'python', 'docker', 'kubernetes'][Math.floor(Math.random() * 4)];
    
    let startTime = Date.now();
    let res = http.get(`${BASE_URL}/api/search?q=${searchTerm}&limit=10`);
    searchLatency.add(Date.now() - startTime);
    
    check(res, {
      'search status is 200': (r) => r.status === 200,
      'search response time < 1s': (r) => r.timings.duration < 1000,
    }) || errorRate.add(1);
  }
  
  // Think time between requests
  sleep(Math.random() * 2 + 1);  // 1-3 seconds
}

export function handleSummary(data) {
  // Custom summary for migration comparison
  const summary = {
    timestamp: new Date().toISOString(),
    metrics: {
      read_p50: data.metrics.read_latency.values['p(50)'],
      read_p95: data.metrics.read_latency.values['p(95)'],
      read_p99: data.metrics.read_latency.values['p(99)'],
      write_p50: data.metrics.write_latency.values['p(50)'],
      write_p95: data.metrics.write_latency.values['p(95)'],
      write_p99: data.metrics.write_latency.values['p(99)'],
      search_p50: data.metrics.search_latency.values['p(50)'],
      search_p95: data.metrics.search_latency.values['p(95)'],
      search_p99: data.metrics.search_latency.values['p(99)'],
      error_rate: data.metrics.errors.values.rate,
      total_requests: data.metrics.http_reqs.values.count,
      rps: data.metrics.http_reqs.values.rate,
    },
    thresholds: {
      passed: Object.values(data.metrics).every(m => !m.thresholds || Object.values(m.thresholds).every(t => t.ok)),
    }
  };
  
  return {
    'stdout': JSON.stringify(summary, null, 2),
    'summary.json': JSON.stringify(summary, null, 2),
  };
}