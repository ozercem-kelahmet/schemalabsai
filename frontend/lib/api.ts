import { API_BASE } from './config'

export const api = {

  health: async () => {
    const res = await fetch(API_BASE + '/api/health')
    return res.json()
  },

  modelInfo: async () => {
    const res = await fetch(API_BASE + '/api/model/info')
    return res.json()
  },

  modelsList: async () => {
    const res = await fetch(API_BASE + '/api/models/list')
    return res.json()
  },

  modelsSwitch: async (modelPath: string) => {
    const res = await fetch(API_BASE + '/api/models/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath })
    })
    return res.json()
  },

  sectors: async () => {
    const res = await fetch(API_BASE + '/api/sectors')
    return res.json()
  },

  predict: async (values: number[][]) => {
    const res = await fetch(API_BASE + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ values })
    })
    return res.json()
  },

  predictSector: async (values: number[][], sector: string) => {
    const res = await fetch(API_BASE + '/api/predict/sector', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ values, sector })
    })
    return res.json()
  },

  upload: async (file: File, folderId?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    if (folderId) formData.append('folder_id', folderId)
    const res = await fetch(API_BASE + '/api/upload', {
      method: 'POST',
      credentials: 'include',
      body: formData
    })
    if (!res.ok) {
      try {
        const data = await res.json()
        throw new Error(data.error || 'Upload failed')
      } catch (e: any) {
        if (e.message && e.message !== 'Upload failed') throw e
        throw new Error('Upload failed')
      }
    }
    return res.json()
  },

  train: async (fileId: string, filename: string, epochs: number = 5, batchSize: number = 64, targetColumn?: string, baseModel?: string) => {
    const res = await fetch(API_BASE + '/api/train', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: fileId, filename, epochs, batch_size: batchSize, target_column: targetColumn, base_model: baseModel })
    })
    return res.json()
  },

  multiTrain: async (fileIds: string[], modelName: string, epochs: number = 5, batchSize: number = 64, learningRate: number = 0.001, warmupSteps: number = 100, queryId?: string, syncMode?: string, scheduleCron?: string, scheduleDesc?: string, connectionIds?: string, selectedTables?: string, baseModel?: string) => {
    const res = await fetch(API_BASE + '/api/train/multi', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_ids: fileIds, model_name: modelName, epochs, batch_size: batchSize, learning_rate: learningRate, warmup_steps: warmupSteps, query_id: queryId, sync_mode: syncMode || "manual", schedule_cron: scheduleCron || "", schedule_desc: scheduleDesc || "", connection_ids: connectionIds || "", selected_tables: selectedTables || "", base_model: baseModel })
    })
    return res.json()
  },

  analyzeFiles: async (fileIds: string[]) => {
    const res = await fetch(API_BASE + '/api/train/analyze', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_ids: fileIds })
    })
    const text = await res.text()
    console.log("analyzeFiles raw response:", text.substring(0, 200))
    try {
      return JSON.parse(text)
    } catch (e) {
      console.error("analyzeFiles JSON parse error:", e, "Response:", text.substring(0, 500))
      throw new Error("analyzeFiles failed: " + text.substring(0, 100))
    }
  },

  getTrainingProgress: async (queryId?: string) => {
    const res = await fetch(API_BASE + '/api/train/progress' + (queryId ? '?query_id=' + queryId : ''), {
      credentials: 'include'
    })
    if (!res.ok) {
      console.error("getTrainingProgress failed:", res.status)
      return { status: "idle", epoch: 0, epochs: 0, accuracy: 0, loss: 0 }
    }
    const text = await res.text()
    try {
      return JSON.parse(text)
    } catch (e) {
      console.error("getTrainingProgress parse error:", text.substring(0, 100))
      return { status: "idle", epoch: 0, epochs: 0, accuracy: 0, loss: 0 }
    }
  },

  getUploadedFiles: async () => {
    const res = await fetch(API_BASE + '/api/files', {
      credentials: 'include'
    })
    return res.json()
  },

  getApiKeys: async () => {
    const res = await fetch(API_BASE + "/api-keys", { credentials: "include" })
    return res.json()
  },
  getQueries: async () => {
    const res = await fetch(API_BASE + "/api/queries", { credentials: "include" })
    return res.json()
  },

  getFineTunedModels: async () => {
    const res = await fetch(API_BASE + '/api/models/finetuned', {
      credentials: 'include'
    })
    return res.json()
  },

  deleteFineTunedModel: async (modelId: string) => {
    const res = await fetch(API_BASE + '/api/models/finetuned/' + modelId, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  renameFineTunedModel: async (modelId: string, newName: string) => {
    const res = await fetch(API_BASE + '/api/models/finetuned/' + modelId, {
      method: 'PATCH',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName })
    })
    return res.json()
  },

  deleteFile: async (fileId: string) => {
    const res = await fetch(API_BASE + '/api/files/delete?id=' + fileId, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  updateFile: async (fileId: string, data: { filename?: string }) => {
    const res = await fetch(API_BASE + "/api/files/update?id=" + fileId, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(data)
    })
    return res.json()
  },


  generateDataset: async (data: {
    name: string
    description: string
    rows: number
    columns: number
    vertical: string
    prompt: string
    use_python: boolean
    python_code: string
  }) => {
    const res = await fetch(API_BASE + "/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(data)
    })
    return res.json()
  },
  getFolders: async () => {
    const res = await fetch(API_BASE + '/api/folders', {
      credentials: 'include'
    })
    return res.json()
  },

  createFolder: async (name: string) => {
    const res = await fetch(API_BASE + '/api/folders/create', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    })
    return res.json()
  },

  updateFolder: async (id: string, name: string) => {
    const res = await fetch(API_BASE + '/api/folders/update?id=' + id, {
      method: 'PUT',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    })
    return res.json()
  },

  deleteFolder: async (id: string) => {
    const res = await fetch(API_BASE + '/api/folders/delete?id=' + id, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  moveFileToFolder: async (fileId: string, folderId: string | null) => {
    const res = await fetch(API_BASE + '/api/files/move', {
      method: 'PUT',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fileId: fileId, folder_id: folderId })
    })
    return res.json()
  },

  chat: async (params: {
    message: string
    file_id: string
    query_id: string
    filename: string
    model: string
    data_context: string
    finetuned_model?: string
    model_path?: string
  }) => {
    const res = await fetch(API_BASE + '/api/chat', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    const text = await res.text()
    let data: any
    try {
      data = JSON.parse(text)
    } catch {
      return { error: text || "Request failed", response: text, status: "error" }
    }
    if (res.status === 403 || data.error) {
      return { error: data.error || data.response || "Request failed", response: data.response || data.error, status: data.status || "error" }
    }
    return data
  },

  chatStream: async (
    params: {
      message: string
      file_id: string
      query_id: string
      filename: string
      model: string
      data_context: string
      finetuned_model?: string
      model_path?: string
    },
    onChunk: (content: string) => void,
    onDone: () => void
  ) => {
    const res = await fetch(API_BASE + '/api/chat', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...params, stream: true })
    })

    // Check for quota/auth errors before streaming
    if (res.status === 403) {
      const errData = await res.json().catch(() => ({ error: "Request failed" }))
      onChunk("⚠️ " + (errData.error || "Quota exceeded"))
      onDone()
      return
    }

    const reader = res.body?.getReader()
    const decoder = new TextDecoder()

    if (!reader) {
      onDone()
      return
    }

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6)
          if (data === '[DONE]') {
            onDone()
            return
          }
          try {
            const parsed = JSON.parse(data)
            if (parsed.choices?.[0]?.delta?.content) {
              onChunk(parsed.choices[0].delta.content)
            }
          } catch {}
        }
      }
    }
    onDone()
  },

  clearChatHistory: async (sessionId: string) => {
    const res = await fetch(API_BASE + '/api/chat/clear', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    })
    return res.json()
  },

  createQuery: async (name: string, model: string, dataSources: string[], fileId?: string, modelName?: string, modelAccuracy?: number, sourceCsvName?: string, trainingModelId?: string) => {
    const res = await fetch(API_BASE + '/api/queries/create', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, model, dataSources, fileId, modelName, modelAccuracy, sourceCsvName, trainingModelId, hasModel: !!modelName })
    })
    return res.json()
  },

  listQueries: async () => {
    const res = await fetch(API_BASE + '/api/queries', {
      credentials: 'include'
    })
    return res.json()
  },

  listDatasets: async () => {
    const res = await fetch(API_BASE + '/api/files', {
      credentials: 'include'
    })
    return res.json()
  },

  deleteQuery: async (id: string) => {
    const res = await fetch(API_BASE + '/api/queries/delete?id=' + id, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  getMessages: async (queryId: string, modelId?: string) => {
    const params = new URLSearchParams()
    if (queryId) params.append('query_id', queryId)
    if (modelId) params.append('model_id', modelId)
    const res = await fetch(API_BASE + '/api/messages?' + params.toString(), {
      credentials: 'include'
    })
    return res.json()
  },

  getQuery: async (queryId: string) => {
    const res = await fetch(API_BASE + '/api/queries/' + queryId, {
      credentials: 'include'
    })
    return res.json()
  },

  getConnections: async () => {
    const res = await fetch(API_BASE + '/api/connections', {
      credentials: 'include'
    })
    return res.json()
  },

  createConnection: async (data: any) => {
    const res = await fetch(API_BASE + '/api/connections/create', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    return res.json()
  },

  deleteConnection: async (id: string) => {
    const res = await fetch(API_BASE + '/api/connections/delete?id=' + id, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  testConnection: async (data: any) => {
    const res = await fetch(API_BASE + '/api/connections/test', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    return res.json()
  },

  listTables: async (connectionId: string, skipCache = false) => {
    const cacheKey = `tables_cache_${connectionId}`
    if (!skipCache) {
      try {
        const cached = sessionStorage.getItem(cacheKey)
        if (cached) {
          const parsed = JSON.parse(cached)
          if (Date.now() - parsed.ts < 30000) return parsed.data  // 30s cache
        }
      } catch {}
    }
    const res = await fetch(API_BASE + '/api/connections/tables?connection_id=' + connectionId, {
      credentials: 'include'
    })
    const data = await res.json()
    try { sessionStorage.setItem(cacheKey, JSON.stringify({ data, ts: Date.now() })) } catch {}
    return data
  },

  exportTable: async (connectionId: string, tableName: string, limit?: number) => {
    const res = await fetch(API_BASE + '/api/connections/export', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ connection_id: connectionId, table_name: tableName, limit: limit || 10000 })
    })
    return res.json()
  }
,

  list: async () => {
    const res = await fetch(API_BASE + '/api/keys', {
      credentials: 'include'
    })
    return res.json()
  },

  create: async (name: string) => {
    const res = await fetch(API_BASE + '/api/keys/create', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    })
    return res.json()
  },

  delete: async (id: string) => {
    const res = await fetch(API_BASE + '/api/keys/delete?id=' + id, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  trainAsync: async (fileIds: string[], modelName: string, epochs: number = 5, batchSize: number = 64, learningRate: number = 0.001, warmupSteps: number = 100, queryId?: string) => {
    const res = await fetch(API_BASE + '/api/train/async', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_ids: fileIds, model_name: modelName, epochs, batch_size: batchSize, learning_rate: learningRate, warmup_steps: warmupSteps, query_id: queryId })
    })
    return res.json()
  },

  getTrainingStatus: async (taskId: string) => {
    const res = await fetch(API_BASE + '/api/train/status?task_id=' + taskId, {
      credentials: 'include'
    })
    return res.json()
  },

  getUploadLimits: async () => {
    const res = await fetch(API_BASE + "/api/config/limits", { credentials: "include" })
    return res.json()
  },
}

