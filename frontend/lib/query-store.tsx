"use client"

import { createContext, useContext, useState, useCallback, ReactNode, useEffect } from "react"

export type QueryItem = {
  id: string
  name: string
  dataSources: string[]
  model: string
  createdAt: Date
  isTraining?: boolean
  hasModel?: boolean
  trainingModelId?: string | null
  fileId?: string
  trainingFailed?: boolean
  modelName?: string
  modelAccuracy?: number
  sourceCsvName?: string
}

type AddQueryInput = {
  id?: string
  name: string
  dataSources?: string[]
  fileId?: string
  model?: string
  isTraining?: boolean
  hasModel?: boolean
  trainingModelId?: string | null
  trainingFailed?: boolean
  skipBackend?: boolean
  modelName?: string
  modelAccuracy?: number
  sourceCsvName?: string
}

type QueryStoreContextType = {
  queries: QueryItem[]
  addQuery: (input: AddQueryInput) => Promise<QueryItem>
  updateQuery: (id: string, updates: Partial<QueryItem>) => void
  deleteQuery: (id: string) => void
  duplicateQuery: (id: string) => Promise<QueryItem | null>
  getQuery: (id: string) => QueryItem | undefined
  isLoaded: boolean
}

const QueryStoreContext = createContext<QueryStoreContextType | null>(null)

export function useQueryStore() {
  const context = useContext(QueryStoreContext)
  if (!context) {
    throw new Error("useQueryStore must be used within QueryStoreProvider")
  }
  return context
}

export function QueryStoreProvider({ children }: { children: ReactNode }) {
  const [queries, setQueries] = useState<QueryItem[]>([])
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    const loadQueries = async () => {
      try {
        console.log("[QueryStore] Loading queries...")
        const res = await fetch("/api/queries", { credentials: "include" })
        if (res.ok) {
          const data = await res.json()
          console.log("[QueryStore] Loaded:", data)
    console.log("[QueryStore] First query fileId:", data.queries[0]?.fileId)
          if (data.queries && Array.isArray(data.queries)) {
            const loaded = data.queries.map((q: any) => ({
              id: q.id,
              name: q.name,
              dataSources: q.dataSources || [],
              model: q.model || "gpt-4o",
              createdAt: new Date(q.createdAt),
              isTraining: q.isTraining || false,
              trainingModelId: q.trainingModelId || null,
              hasModel: q.hasModel || false,
              trainingFailed: q.trainingFailed || false
            }))
            setQueries(loaded)
          }
        }
      } catch (e) {
        console.error("[QueryStore] Failed to load queries:", e)
      }
      setIsLoaded(true)
    }
    loadQueries()
  }, [])

  const addQuery = useCallback(async (input: AddQueryInput): Promise<QueryItem> => {
    console.log("[QueryStore] addQuery called with:", input)
    
    const tempId = "temp-" + Date.now()
    const newQuery: QueryItem = {
      id: input.id || tempId,
      name: input.name,
      dataSources: input.dataSources || [],
      model: input.model || "gpt-4o",
      createdAt: new Date(),
      isTraining: input.isTraining || false,
      trainingModelId: input.trainingModelId || null,
      hasModel: input.hasModel || false,
      modelName: input.modelName || '',
      modelAccuracy: input.modelAccuracy || 0,
      sourceCsvName: input.sourceCsvName || '',
    }
    
    // Add to state immediately for UI
    setQueries(prev => [newQuery, ...prev])

    // If ID is provided or skipBackend, don't call backend
    if (input.id || input.skipBackend) {
      console.log("[QueryStore] Skipping backend")
      return newQuery
    }

    // Create on backend and wait for real ID
    try {
      console.log("[QueryStore] Sending to backend...")
      const res = await fetch("/api/queries/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          name: newQuery.name,
          model: newQuery.model,
          dataSources: newQuery.dataSources,
          fileId: input.fileId || "",
          isTraining: newQuery.isTraining,
          hasModel: newQuery.hasModel,
          trainingModelId: newQuery.trainingModelId,
          modelName: newQuery.modelName,
          modelAccuracy: newQuery.modelAccuracy,
          sourceCsvName: newQuery.sourceCsvName,
        }),
      })
      
      const data = await res.json()
      console.log("[QueryStore] Response:", data)
      
      if (data.id) {
        // Update temp ID with real ID and sync isTraining from backend
        newQuery.id = data.id
        newQuery.isTraining = data.isTraining ?? newQuery.isTraining
        setQueries(prev => prev.map(q => 
          q.id === tempId ? { ...q, id: data.id, isTraining: data.isTraining ?? q.isTraining } : q
        ))
      }
    } catch (e) {
      console.error("[QueryStore] Failed to save query:", e)
    }

    return newQuery
  }, [])

  const updateQuery = useCallback((id: string, updates: Partial<QueryItem>) => {
    console.log("[QueryStore] updateQuery:", id, updates)
    setQueries(prev => prev.map(q => q.id === id ? { ...q, ...updates } : q))

    fetch("/api/queries/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ id, ...updates }),
    }).catch(e => console.error("[QueryStore] Failed to update query:", e))
  }, [])

  const deleteQuery = useCallback((id: string) => {
    console.log("[QueryStore] deleteQuery:", id)
    setQueries(prev => prev.filter(q => q.id !== id))

    fetch(`/api/queries/delete?id=${id}`, {
      method: "DELETE",
      credentials: "include",
    }).catch(e => console.error("[QueryStore] Failed to delete query:", e))
  }, [])

  const duplicateQuery = useCallback(async (id: string): Promise<QueryItem | null> => {
    const original = queries.find(q => q.id === id)
    if (!original) return null

    return addQuery({
      name: original.name + " (copy)",
      dataSources: original.dataSources,
      model: original.model,
    })
  }, [queries, addQuery])

  const getQuery = useCallback((id: string): QueryItem | undefined => {
    return queries.find(q => q.id === id)
  }, [queries])

  return (
    <QueryStoreContext.Provider value={{ queries, addQuery, updateQuery, deleteQuery, duplicateQuery, getQuery, isLoaded }}>
      {children}
    </QueryStoreContext.Provider>
  )
}
