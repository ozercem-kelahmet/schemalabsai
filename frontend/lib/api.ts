import { API_BASE } from './config'

export const api = {
  health: async () => {
    const res = await fetch(API_BASE + '/health')
    return res.json()
  },

  modelInfo: async () => {
    const res = await fetch(API_BASE + '/model/info')
    return res.json()
  },

  modelsList: async () => {
    const res = await fetch(API_BASE + '/models/list')
    return res.json()
  },

  modelsSwitch: async (modelPath: string) => {
    const res = await fetch(API_BASE + '/models/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath })
    })
    return res.json()
  },

  sectors: async () => {
    const res = await fetch(API_BASE + '/sectors')
    return res.json()
  },

  predict: async (values: number[][]) => {
    const res = await fetch(API_BASE + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ values })
    })
    return res.json()
  },

  predictSector: async (values: number[][], sector: string) => {
    const res = await fetch(API_BASE + '/predict/sector', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ values, sector })
    })
    return res.json()
  },

  upload: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const res = await fetch(API_BASE + '/upload', {
      method: 'POST',
      credentials: 'include',
      body: formData
    })
    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || 'Upload failed')
    }
    return res.json()
  },

  train: async (fileId: string, filename: string, epochs: number = 5, batchSize: number = 64, targetColumn?: string) => {
    const res = await fetch(API_BASE + '/train', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: fileId, filename, epochs, batch_size: batchSize, target_column: targetColumn })
    })
    return res.json()
  },

  multiTrain: async (fileIds: string[], modelName: string, epochs: number = 5, batchSize: number = 64, learningRate: number = 0.001, warmupSteps: number = 100, queryId?: string) => {
    const res = await fetch(API_BASE + '/train/multi', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_ids: fileIds, model_name: modelName, epochs, batch_size: batchSize, learning_rate: learningRate, warmup_steps: warmupSteps, query_id: queryId })
    })
    return res.json()
  },

  analyzeFiles: async (fileIds: string[]) => {
    const res = await fetch(API_BASE + '/train/analyze', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_ids: fileIds })
    })
    return res.json()
  },

  getTrainingProgress: async (queryId?: string) => {
    const res = await fetch(API_BASE + '/train/progress' + (queryId ? '?query_id=' + queryId : ''), {
      credentials: 'include'
    })
    return res.json()
  },

  getUploadedFiles: async () => {
    const res = await fetch(API_BASE + '/files', {
      credentials: 'include'
    })
    return res.json()
  },

  getApiKeys: async () => {
    const res = await fetch(API_BASE + "/api-keys", { credentials: "include" })
    return res.json()
  },
  getQueries: async () => {
    const res = await fetch(API_BASE + "/queries", { credentials: "include" })
    return res.json()
  },

  getFineTunedModels: async () => {
    const res = await fetch(API_BASE + '/models/finetuned', {
      credentials: 'include'
    })
    return res.json()
  },

  deleteFineTunedModel: async (modelId: string) => {
    const res = await fetch(API_BASE + '/models/finetuned/' + modelId, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  renameFineTunedModel: async (modelId: string, newName: string) => {
    const res = await fetch(API_BASE + '/models/finetuned/' + modelId, {
      method: 'PATCH',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName })
    })
    return res.json()
  },

  deleteFile: async (fileId: string) => {
    const res = await fetch(API_BASE + '/files/delete?id=' + fileId, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  getFolders: async () => {
    const res = await fetch(API_BASE + '/folders', {
      credentials: 'include'
    })
    return res.json()
  },

  createFolder: async (name: string) => {
    const res = await fetch(API_BASE + '/folders/create', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    })
    return res.json()
  },

  updateFolder: async (id: string, name: string) => {
    const res = await fetch(API_BASE + '/folders/update?id=' + id, {
      method: 'PUT',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    })
    return res.json()
  },

  deleteFolder: async (id: string) => {
    const res = await fetch(API_BASE + '/folders/delete?id=' + id, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  moveFileToFolder: async (fileId: string, folderId: string | null) => {
    const res = await fetch(API_BASE + '/files/move', {
      method: 'PUT',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: fileId, folder_id: folderId })
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
  }) => {
    const res = await fetch(API_BASE + '/chat', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    return res.json()
  },

  chatStream: async (
    params: {
      message: string
      file_id: string
      query_id: string
      filename: string
      model: string
      data_context: string
    },
    onChunk: (content: string) => void,
    onDone: () => void
  ) => {
    const res = await fetch(API_BASE + '/chat', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...params, stream: true })
    })

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
    const res = await fetch(API_BASE + '/chat/clear', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    })
    return res.json()
  },

  createQuery: async (name: string, model: string, dataSources: string[]) => {
    const res = await fetch(API_BASE + '/queries/create', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, model, data_sources: dataSources })
    })
    return res.json()
  },

  listQueries: async () => {
    const res = await fetch(API_BASE + '/queries', {
      credentials: 'include'
    })
    return res.json()
  },

  deleteQuery: async (id: string) => {
    const res = await fetch(API_BASE + '/queries/delete?id=' + id, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  getMessages: async (queryId: string) => {
    const res = await fetch(API_BASE + '/messages?query_id=' + queryId, {
      credentials: 'include'
    })
    return res.json()
  },

  getConnections: async () => {
    const res = await fetch(API_BASE + '/connections', {
      credentials: 'include'
    })
    return res.json()
  },

  createConnection: async (data: any) => {
    const res = await fetch(API_BASE + '/connections/create', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    return res.json()
  },

  deleteConnection: async (id: string) => {
    const res = await fetch(API_BASE + '/connections/delete?id=' + id, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  },

  testConnection: async (data: any) => {
    const res = await fetch(API_BASE + '/connections/test', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    return res.json()
  },

  listTables: async (connectionId: string) => {
    const res = await fetch(API_BASE + '/connections/tables?connection_id=' + connectionId, {
      credentials: 'include'
    })
    return res.json()
  },

  exportTable: async (connectionId: string, tableName: string, limit?: number) => {
    const res = await fetch(API_BASE + '/connections/export', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ connection_id: connectionId, table_name: tableName, limit: limit || 10000 })
    })
    return res.json()
  }
}

// API Keys
export const apiKeys = {
  list: async () => {
    const res = await fetch(API_BASE + '/keys', {
      credentials: 'include'
    })
    return res.json()
  },

  create: async (name: string) => {
    const res = await fetch(API_BASE + '/keys/create', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    })
    return res.json()
  },

  delete: async (id: string) => {
    const res = await fetch(API_BASE + '/keys/delete?id=' + id, {
      method: 'DELETE',
      credentials: 'include'
    })
    return res.json()
  }
}
