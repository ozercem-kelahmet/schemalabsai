// API Base URL - works for both local and production
export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 
  (typeof window !== 'undefined' 
    ? `${window.location.origin}/api`
    : 'http://localhost:8080/api')

export const API_V1_BASE = process.env.NEXT_PUBLIC_API_URL 
  ? process.env.NEXT_PUBLIC_API_URL.replace('/api', '/v1')
  : (typeof window !== 'undefined' 
    ? `${window.location.origin}/v1`
    : 'http://localhost:8080/v1')
