const isProduction = typeof window !== 'undefined' && window.location.hostname !== 'localhost'

export const API_BASE = isProduction ? '' : 'http://localhost:8080'
