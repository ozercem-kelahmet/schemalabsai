// Core types for SchemaLabs

export type DataSource = "uploaded-files" | "databricks" | "supabase" | "api" | "google-drive" | "postgresql" | "mysql" | "mongodb" | "snowflake" | "pinecone" | "weaviate" | "chroma" | "lancedb" | "gcs" | "aws-s3" | "graphql" | "rest" | "upload" | "generated"
export type Vertical = "finance" | "healthcare" | "e-commerce" | "marketing" | "hr" | "operations"
export type Complexity = "simple" | "medium" | "advanced"
export type RowCount = "small" | "medium" | "large"

export type SyncMode = "real-time" | "scheduled" | "manual"

export type ConnectionType = "upload" | "api" | "database"
export type ApiType = "rest" | "graphql"
export type DatabaseType = "postgresql" | "supabase" | "mongodb" | "databricks" | "snowflake" | "pinecone"
export type CloudStorageType = "google-drive" | "gcs" | "aws-s3"

export interface DataConnection {
  id: string
  name: string
  type: ConnectionType
  subType?: ApiType | DatabaseType | CloudStorageType
  endpoint?: string
  authToken?: string
  apiKey?: string
  environment?: string
  bucket?: string
  region?: string
  projectId?: string
  createdAt: Date
  status: "connected" | "disconnected" | "error"
}

export interface DataFolder {
  id: string
  name: string
  datasetIds: string[]
  createdAt: Date
}

export interface Dataset {
  id: string
  name: string
  description: string
  source: DataSource
  vertical: Vertical
  complexity: Complexity
  rowCount: RowCount
  rows: number
  columns: number
  schema: ColumnSchema[]
  sampleData: Record<string, unknown>[]
  lastUpdated?: Date
  syncStatus?: "synced" | "pending" | "outdated"
  folderId?: string
  connectionId?: string
}

export interface ColumnSchema {
  name: string
  type: "string" | "number" | "boolean" | "date"
  description?: string
}

export interface Model {
  id: string
  name: string
  description: string
  datasets: DatasetConnection[]
  syncMode: SyncMode
  baseModel: string
  accuracy: number
  createdAt: Date
  updatedAt: Date
  status: "training" | "completed" | "failed" | "needs-update"
  pendingUpdates?: DatasetUpdate[]
  // New fields for tracking usage and metrics
  modelId: string // Unique model identifier for API
  apiRequests: number
  tokensUsed: number
  trainingMetricsHistory?: {
    epoch: number
    loss: number
    accuracy: number
  }[]
  endpoints?: ModelEndpoint[]
}

export interface ModelEndpoint {
  id: string
  name: string
  urlPath: string
  description: string
  createdAt: Date
  status: "active" | "inactive"
}

export interface DatasetConnection {
  datasetId: string
  datasetName: string
  source: DataSource
  rows: number
  columns: number
  connectedAt: Date
  lastSynced: Date
  syncStatus: "synced" | "pending" | "outdated"
}

export interface DatasetUpdate {
  datasetId: string
  datasetName: string
  updateType: "modified" | "deleted" | "schema-changed"
  detectedAt: Date
  message: string
}

export interface TrainingMetrics {
  epoch: number
  totalEpochs: number
  loss: number
  accuracy: number
  learningRate: number
}

export interface EvaluationMetrics {
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  confusionMatrix?: number[][]
}

// Playground types
export type LLMProvider = "claude" | "gpt-4" | "gemini" | "llama"

export interface LLMOption {
  id: LLMProvider
  name: string
  version: string
  provider: string
}

export interface PlaygroundMessage {
  id: string
  role: "user" | "assistant"
  content: string
  modelId?: string
  llmId?: LLMProvider
  timestamp: Date
  prediction?: {
    result: string
    confidence: number
    details?: Record<string, unknown>
  }
}

export interface PlaygroundSession {
  id: string
  name: string
  modelIds: string[]
  llmIds: LLMProvider[]
  messages: PlaygroundMessage[]
  createdAt: Date
  updatedAt: Date
}
