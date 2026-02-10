"use client"

import { create } from "zustand"

export interface SharedEndpoint {
  id: string
  name: string
  url: string
  modelId: string
  modelName: string
  description: string
  status: "active" | "inactive"
  requests: number
  createdAt: Date
}

interface EndpointsState {
  endpoints: SharedEndpoint[]
  addEndpoint: (endpoint: SharedEndpoint) => void
  removeEndpoint: (id: string) => void
  updateEndpoint: (id: string, updates: Partial<SharedEndpoint>) => void
}

// Initial mock endpoints
const initialEndpoints: SharedEndpoint[] = [
  {
    id: "ep-1",
    name: "Churn Prediction",
    url: "/v1/models/cust-intl/predict",
    modelId: "model-001",
    modelName: "Customer Intelligence",
    description: "Predict customer churn probability",
    status: "active",
    requests: 12847,
    createdAt: new Date("2024-01-16"),
  },
  {
    id: "ep-2",
    name: "Risk Assessment",
    url: "/v1/models/fin-risk/assess",
    modelId: "model-002",
    modelName: "Financial Risk Model",
    description: "Assess financial risk for loan applications",
    status: "active",
    requests: 8432,
    createdAt: new Date("2024-01-14"),
  },
]

export const useEndpointsStore = create<EndpointsState>((set) => ({
  endpoints: initialEndpoints,
  addEndpoint: (endpoint) =>
    set((state) => ({
      endpoints: [...state.endpoints, endpoint],
    })),
  removeEndpoint: (id) =>
    set((state) => ({
      endpoints: state.endpoints.filter((ep) => ep.id !== id),
    })),
  updateEndpoint: (id, updates) =>
    set((state) => ({
      endpoints: state.endpoints.map((ep) =>
        ep.id === id ? { ...ep, ...updates } : ep
      ),
    })),
}))
