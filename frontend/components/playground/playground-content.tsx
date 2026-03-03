"use client"
import { toast } from "sonner"

import type React from "react"
import { useState, useEffect, useRef, useCallback, useMemo } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Textarea } from "@/components/ui/textarea"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import { SourceBadge } from "@/components/datasets/source-badge"
import { useSidebar } from "@/components/layout/sidebar"
import { useQueryStore } from "@/lib/query-store"
import type { LLMProvider, DataSource } from "@/lib/types"
import {
  Loader2,
  User,
  Database,
  ChevronDown,
  Clock,
  ArrowUp,
  ArrowDown,
  Box,
  Zap,
  Check,
  ChevronLeft,
  ChevronRight,
  Search,
  GitCompare,
  Plus,
  Settings2,
  Sparkles,
} from "lucide-react"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { ContentRenderer, FunctionCallDisplay, type ResponseBlock } from "./response-renderer"
import { VerticalPanel } from "./vertical-panel"
import { api } from "@/lib/api"

const defaultLLMOptions = [
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", provider: "Anthropic" },
  { id: "claude-opus-4", name: "Claude Opus 4", provider: "Anthropic" },
  { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "OpenAI" },
  { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", provider: "Google" },
  { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "Google" },
]

interface BackendModel {
  id: string
  name: string
  accuracy?: number
  sourceCsvName?: string
  sourceFiles?: string
  source_files?: string
  source_csv_name?: string
  source_file_names?: string
  source_file_id?: string
  sourceFileId?: string
  model_path?: string
  modelPath?: string
}

interface AdaptedModel {
  id: string
  name: string
  accuracy: number
  datasets: { datasetId: string; datasetName: string; source: DataSource }[]
  sourceFiles?: string
  modelPath?: string
}

interface DisplayMessage {
  id: string
  role: "user" | "assistant"
  content: string
  model?: string
  modelId?: string
  llmId?: string
  tokens?: number
  time?: string
  timestamp: Date
  isLoading?: boolean
  groupId?: string
  functionCalls?: FunctionCallInfo[]
}

interface FunctionCallInfo {
  function_name: string
  arguments: any
  result: any
  error?: string
  execution_ms: number
}

function adaptBackendModel(m: BackendModel): AdaptedModel {
  const sourceFilesStr = m.sourceFiles || m.source_files || ""
  const sourceCsvName = m.sourceCsvName || m.source_csv_name || ""
  const sourceFileId = m.source_file_id || m.sourceFileId || ""
  const sourceFileNames = (m.source_file_names || "").split(",").filter(Boolean)
  const sourceFiles = sourceFilesStr ? sourceFilesStr.split(",").filter(Boolean) : []
  
  let datasets: { datasetId: string; datasetName: string; source: DataSource }[] = []
  
  if (sourceFiles.length > 0) {
    datasets = sourceFiles.map((file, idx) => ({
      datasetId: file.trim(),
      datasetName: (() => {
        const raw = (sourceFileNames[idx] || file).trim()
        if (raw.startsWith("conn_")) {
          // Connection file: use filename from uploaded_files or extract table name
          const parts = raw.replace(/\.csv$/, "").split("_")
          return parts.length >= 3 ? parts.slice(2).join("_") : raw
        }
        return raw.replace(/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}[_.]?/, "").replace(/_\d{8}_\d{6}/, "").replace(/\.csv$/, "")
      })(),
      source: (file.trim().startsWith("conn_") ? "connection" : "upload") as DataSource,
    }))
  } else if (sourceFileId) {
    datasets = [{ datasetId: sourceFileId, datasetName: sourceCsvName || sourceFileId, source: "upload" as DataSource }]
  } else if (sourceCsvName) {
    datasets = [{ datasetId: m.id, datasetName: sourceCsvName, source: "upload" as DataSource }]
  } else if (m.source_name && m.source_name !== "0 files merged") {
    datasets = [{ datasetId: m.id, datasetName: m.source_name, source: "upload" as DataSource }]
  }
  
  const connectionIds = m.connection_ids || m.connectionIds || ""
  const isConnectionBased = connectionIds !== ""

  return {
    id: m.id,
    name: m.name || "Unnamed Model",
    accuracy: m.accuracy || 0,
    datasets,
    sourceFiles: sourceFilesStr,
    modelPath: m.model_path || m.modelPath || "",
    isConnectionBased,
    connectionName: m.connection_names || m.connectionNames || "",
  }
}

const MODELS_PER_PAGE = 12

interface PlaygroundContentProps {
  sessionId?: string
}

export function PlaygroundContent({ sessionId: propSessionId }: PlaygroundContentProps = {}) {
  const searchParams = useSearchParams()
  const router = useRouter()
  const querySessionId = searchParams.get("session")
  const sessionId = propSessionId || querySessionId
  const newChatTrigger = searchParams.get("new")
  const modelIdFromUrl = searchParams.get("model")
  const autoMessage = searchParams.get("autoMessage")
  const { chatSessions, setChatSessions } = useSidebar()
  const { queries, getQuery } = useQueryStore()
  const currentQuery = useMemo(() => {
    const q = sessionId ? getQuery(sessionId) : null
    return q
  }, [sessionId, getQuery, queries])

  // Models state
  const [backendModels, setBackendModels] = useState<AdaptedModel[]>([])
  const [modelsLoading, setModelsLoading] = useState(true)
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [selectedFiles, setSelectedFiles] = useState<any[]>([])
  const [currentPage, setCurrentPage] = useState(0)
  const [selectedModels, setSelectedModels] = useState<AdaptedModel[]>([])
  const [compareMode, setCompareMode] = useState(false)
  const selectedModel = compareMode ? selectedModels[0] || null : selectedModels[0] || null
  const [verticalPanelOpen, setVerticalPanelOpen] = useState(false)

  // Chat state
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [currentQueryId, setCurrentQueryId] = useState<string | null>(null)
  const [llmOptions, setLlmOptions] = useState(defaultLLMOptions)
  const [keyStatus, setKeyStatus] = useState<Record<string, boolean>>({})
  const [addKeyModal, setAddKeyModal] = useState<{open: boolean; provider: string; providerLabel: string}>({open: false, provider: "", providerLabel: ""})
  const [addKeyValue, setAddKeyValue] = useState("")
  const [addKeySaving, setAddKeySaving] = useState(false)
  const [selectedLLMs, setSelectedLLMs] = useState<string[]>([])

  // Fetch available LLM models
  useEffect(() => {
    fetch("/api/vertical/llm/models", { credentials: "include" })
      .then(r => r.json())
      .then((models: {id: string; name: string; provider: string}[]) => {
        if (models && models.length > 0) {
          setLlmOptions(models.map(m => ({ id: m.id, name: m.name, provider: m.provider })))
          // Set default LLM - only keep models that are visible in dropdown
          const schemaModels = models.filter(m => m.provider === "Schema")
          const defaultId = schemaModels.length > 0 ? schemaModels[0].id : ""
          // Force reset to Schema model - ignore any restored non-visible models
          setTimeout(() => {
            setSelectedLLMs(prev => {
              const filtered = prev.filter(id => {
                const model = models.find(m => m.id === id)
                if (!model) return false
                if (model.provider === "Schema") return true
                // Check keyStatus for other providers
                return false
              })
              if (filtered.length > 0) return filtered
              return defaultId ? [defaultId] : []
            })
          }, 500)
        }
      })
      .catch(() => {})
    fetch("/api/vertical/llm/key-status", { credentials: "include" })
      .then(r => r.json())
      .then(status => {
        if (status) setKeyStatus(status)
      })
      .catch(() => {})
  }, [])

  // Silent refresh state
  const [hasInitializedChat, setHasInitializedChat] = useState(false)
  const [refreshCount, setRefreshCount] = useState(0)

  // Client mount state
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])

  // Scroll state
  const [showScrollButton, setShowScrollButton] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  // UI state
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const [llmDropdownOpen, setLlmDropdownOpen] = useState(false)
  const [modelSearchQuery, setModelSearchQuery] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Fetch models from backend (with cache for instant display)
  useEffect(() => {
    // Load from cache immediately + restore last selected model
    try {
      const cached = localStorage.getItem("schemalabs_models_cache")
      if (cached) {
        const parsed = JSON.parse(cached)
        if (parsed.models?.length > 0) {
          const cachedModels = parsed.models.map(adaptBackendModel)
          setBackendModels(cachedModels)
          setModelsLoading(false)
          // Restore model from URL param or session cache
          const urlModel = new URLSearchParams(window.location.search).get("model")
          const targetModelId = urlModel || (sessionId ? localStorage.getItem(`schemalabs_session_model_${sessionId}`) : null)
          if (targetModelId) {
            const model = cachedModels.find((m: any) => m.id === targetModelId || m.id?.includes(targetModelId))
            if (model) {
              setSelectedModels([model])
              setHasInitializedChat(true)
              if (model.datasets?.length > 0) {
                setSelectedFiles(model.datasets.map((ds: any) => ({ file_id: ds.datasetId, filename: ds.datasetName, source: ds.source })))
              }
            }
          }
        }
      }
    } catch {}
    
    const loadAll = async () => {
      // Load models and files first (fast), queries in background (slow)
      const [modelsRes, filesRes, messagesRes] = await Promise.all([
        api.getFineTunedModels().catch(() => ({ models: [] })),
        api.getUploadedFiles().catch(() => ({ files: [] })),
        sessionId ? api.getMessages(sessionId).catch(() => ({ messages: [] })) : Promise.resolve(null),
      ])
      const queriesPromise = api.getQueries().catch(() => ({ queries: [] }))
      const allModels = modelsRes.models && Array.isArray(modelsRes.models) ? modelsRes.models.map(adaptBackendModel) : []
      setBackendModels(allModels)
      setModelsLoading(false)
      try { localStorage.setItem("schemalabs_models_cache", JSON.stringify({ models: modelsRes.models })) } catch {}
      if (filesRes.files) {
        setUploadedFiles(filesRes.files)
      }
      // Set model immediately from messages
      let modelSetFromMessages = false
      if (messagesRes?.messages?.length > 0 && allModels.length > 0) {
        const modelIds = [...new Set(messagesRes.messages.filter((m: any) => m.role === "assistant" && m.finetuned_model_id).map((m: any) => m.finetuned_model_id))]
        if (modelIds.length > 0) {
          const models = modelIds.map((id: string) => allModels.find(m => m.id === id)).filter(Boolean)
          if (models.length > 0) {
            setSelectedModels(models as any)
            modelSetFromMessages = true
            if ((models[0] as any)?.datasets?.length > 0) {
              setSelectedFiles((models[0] as any).datasets.map((ds: any) => ({ file_id: ds.datasetId, filename: ds.datasetName, source: ds.source })))
            }
          }
        }
        const llmMsg = messagesRes.messages.find((m: any) => m.role === "assistant" && m.model)
        if (llmMsg?.model) {
          const ml = llmMsg.model.toLowerCase()
          if (llmOptions.length === 0 || llmOptions.some(l => l.id === llmMsg.model && (l.provider === "Schema" || keyStatus.unlimited || keyStatus[l.provider === "Google" ? "gemini" : l.provider.toLowerCase()]))) setSelectedLLMs([llmMsg.model])
        }
      }
      // Fallback: if model not set from messages, try from query
      const queriesRes = await queriesPromise
      if (!modelSetFromMessages && sessionId && allModels.length > 0) {
        const queryList = queriesRes?.queries || queriesRes || []
        const thisQuery = Array.isArray(queryList) ? queryList.find((q: any) => q.id === sessionId) : null
        if (thisQuery?.trainingModelId) {
          const tid = thisQuery.trainingModelId
          let model = allModels.find(m => m.id === tid || m.name === tid || m.modelPath === tid)
          // Fallback: match by date pattern in model path
          if (!model) {
            const dateMatch = tid.match(/(\d{8})/)
            if (dateMatch) {
              model = allModels.find(m => m.modelPath?.includes(dateMatch[1]) || m.name?.includes(dateMatch[1]))
            }
          }
          if (model) {
            setSelectedModels([model] as any)
            modelSetFromMessages = true
            if ((model as any)?.datasets?.length > 0) {
              setSelectedFiles((model as any).datasets.map((ds: any) => ({ file_id: ds.datasetId, filename: ds.datasetName, source: ds.source })))
            }
          }
        }
        // Last fallback: try modelName from query
        if (!modelSetFromMessages && thisQuery?.modelName) {
          const model = allModels.find(m => m.name === thisQuery.modelName)
          if (model) {
            setSelectedModels([model] as any)
            modelSetFromMessages = true
            if ((model as any)?.datasets?.length > 0) {
              setSelectedFiles((model as any).datasets.map((ds: any) => ({ file_id: ds.datasetId, filename: ds.datasetName, source: ds.source })))
            }
          }
        }
      }
      // If coming from build page with model param, set model immediately
      if (modelIdFromUrl && allModels.length > 0 && !modelSetFromMessages) {
        const urlModel = allModels.find(m => m.id === modelIdFromUrl || m.id?.includes(modelIdFromUrl))
        if (urlModel) {
          setSelectedModels([urlModel] as any)
          setHasInitializedChat(true)
          if ((urlModel as any)?.datasets?.length > 0) {
            setSelectedFiles((urlModel as any).datasets.map((ds: any) => ({ file_id: ds.datasetId, filename: ds.datasetName, source: ds.source })))
          }
        }
      }
    }
    loadAll()
  }, [])

  // When backendModels load, set model and LLM from existing messages
  useEffect(() => {
    if (backendModels.length === 0 || selectedModels.length > 0) return
    const assistantMsgs = messages.filter(m => m.role === "assistant" && m.modelId)
    const modelIds = [...new Set(assistantMsgs.map(m => m.modelId))]
    if (modelIds.length > 0) {
      const models = modelIds.map(id => backendModels.find(m => m.id === id)).filter(Boolean) as typeof backendModels
      if (models.length > 0) {
        setSelectedModels(models)
        if (models[0].datasets?.length > 0) {
          setSelectedFiles(models[0].datasets.map(ds => ({
            file_id: ds.datasetId,
            filename: ds.datasetName,
            source: ds.source
          })))
        }
        // Cache model for this session
        if (sessionId) {
          try { localStorage.setItem(`schemalabs_session_model_${sessionId}`, models[0].id) } catch {}
        }
      }
    }
    // Also set LLM
    const llmMsg = messages.find(m => m.role === "assistant" && m.model)
    if (llmMsg?.model) {
      const ml = llmMsg.model.toLowerCase()
      if (llmOptions.length === 0 || llmOptions.some(l => l.id === llmMsg.model && (l.provider === "Schema" || keyStatus.unlimited || keyStatus[l.provider === "Google" ? "gemini" : l.provider.toLowerCase()]))) setSelectedLLMs([llmMsg.model])
    }
  }, [backendModels, messages])

  // Select model from URL parameter
  useEffect(() => {
    if (!modelIdFromUrl || backendModels.length === 0) return
    const model = backendModels.find(m => m.id === modelIdFromUrl || m.id?.includes(modelIdFromUrl || ""))
    if (model) {
      setHasInitializedChat(true)
      setSelectedModels([model])
      if (model.datasets && model.datasets.length > 0) {
        setSelectedFiles(model.datasets.map(ds => ({
          file_id: ds.datasetId,
          filename: ds.datasetName,
          source: ds.source
        })))
      }
      // Auto-send message from build page
      if (false) {
        setTimeout(() => {
          setInput(autoMessage)
          const submitBtn = document.querySelector('[data-send-button]') as HTMLButtonElement
          if (submitBtn) submitBtn.click()
        }, 500)
      }
    }
  }, [modelIdFromUrl, backendModels])

  // Set selected model when currentQuery or backendModels changes
  useEffect(() => {
    if (!currentQuery || backendModels.length === 0) return
    
    // Try to find model by trainingModelId
    if (currentQuery.trainingModelId) {
      const trainingId = currentQuery.trainingModelId
      // Try exact match first, then partial match by date
      let model = backendModels.find(m => m.id === trainingId || m.name === trainingId || m.modelPath === trainingId)
      if (!model) {
        // Try matching by date pattern (e.g., 20260122)
        const dateMatch = trainingId.match(/(\d{8})/)
        if (dateMatch) {
          model = backendModels.find(m => m.name?.includes(dateMatch[1]) || m.modelPath?.includes(dateMatch[1]))
        }
      }
      if (model) {
      setHasInitializedChat(true)
        setSelectedModels([model])
        // Also set files
        if (model.datasets && model.datasets.length > 0) {
          setSelectedFiles(model.datasets.map(ds => ({
            file_id: ds.datasetId,
            filename: ds.datasetName,
            source: ds.source
          })))
        }
        return
      }
    }
    
    // Try by query name
    if (currentQuery.name) {
      const model = backendModels.find(m => m.name === currentQuery.name)
      if (model) {
      setHasInitializedChat(true)
        setSelectedModels([model])
        if (model.datasets && model.datasets.length > 0) {
          setSelectedFiles(model.datasets.map(ds => ({
            file_id: ds.datasetId,
            filename: ds.datasetName,
            source: ds.source
          })))
        }
      }
    }
  }, [currentQuery, backendModels])

  // Set selected files when currentQuery changes
  useEffect(() => {
    if (!currentQuery || uploadedFiles.length === 0) return
    
    // Skip if model already set files from datasets
    if (selectedModel && selectedModel.datasets && selectedModel.datasets.length > 0) return
    
    let files: any[] = []
    
    // Check sourceFiles first
    if ((currentQuery as any).sourceFiles) {
      const sourceFileIds = (currentQuery as any).sourceFiles.split(",")
      files = sourceFileIds
        .map((id: string) => uploadedFiles.find((f: any) => f.file_id === id || f.file_id === id + ".csv"))
        .filter(Boolean)
    }
    
    // Then check dataSources
    if (files.length === 0 && currentQuery.dataSources) {
      files = currentQuery.dataSources
        .map((id: string) => uploadedFiles.find((f: any) => f.file_id === id))
        .filter(Boolean)
    }
    
    // Fallback
    if (files.length === 0 && uploadedFiles.length > 0) {
      files = [uploadedFiles[uploadedFiles.length - 1]]
    }
    
    setSelectedFiles(files)
  }, [currentQuery, uploadedFiles])

  // Update currentQueryId when session changes
  useEffect(() => {
    if (sessionId && sessionId !== currentQueryId) {
      // Reset for new session
      setHasInitializedChat(false)
      setMessages([])
      setCurrentQueryId(sessionId)
    }
  }, [sessionId, currentQueryId])

  // Load messages when session changes
  useEffect(() => {
    if (sessionId && !hasInitializedChat) {
      const session = chatSessions.find(s => s.id === sessionId)
      if (session && session.modelIds?.length > 0) {
        const model = backendModels.find(m => m.id === session.modelIds[0])
        if (model) setSelectedModels([model])
      }
      setCurrentQueryId(sessionId)
      // setMessagesLoading(true) - disabled for silent refresh
      
api.getMessages(sessionId)
        .then(data => {
          if (data.messages && Array.isArray(data.messages)) {
            // Group assistant messages by similar timestamp (within 5 seconds)
            const rawMessages = data.messages.map((m: any) => ({
              id: m.id,
              role: m.role,
              content: m.content,
              model: m.model,
              modelId: m.finetuned_model_id || "",
              compareGroup: m.compare_group || "",
              tokens: m.tokens,
              time: m.time_taken || "",
              timestamp: new Date(m.created_at),
              functionCalls: m.function_calls ? (typeof m.function_calls === "string" ? JSON.parse(m.function_calls) : m.function_calls) : [],
            }))
            
            console.log("RAW MESSAGES:", rawMessages.map((m: any, i: number) => `${i}:${m.role}|${m.model||""}|${m.modelId?.slice(0,8)||""}|${m.content?.slice(0,30)}`))
            
            // Assign groupIds - use compare_group if available, else pattern matching
            const loadedMessages: DisplayMessage[] = []
            const compareGroups = new Map<string, string>()
            
            // Group by compare_group when available
            // For legacy messages (no compare_group): each user starts a new group
            const cgMap = new Map<string, string>() // compare_group -> groupId
            let currentLegacyGroup: string | null = null
            
            rawMessages.forEach((m: any, idx: number) => {
              const cg = m.compareGroup || ""
              
              if (m.role === "user") {
                if (cg && cgMap.has(cg)) {
                  // Duplicate user in same compare group - skip
                  return
                }
                const groupId = cg ? `group-${cg}` : `group-${m.id || idx}`
                if (cg) cgMap.set(cg, groupId)
                currentLegacyGroup = groupId
                loadedMessages.push({ ...m, groupId })
              } else {
                // Assistant
                if (cg && cgMap.has(cg)) {
                  loadedMessages.push({ ...m, groupId: cgMap.get(cg)! })
                } else if (cg) {
                  const groupId = `group-${cg}`
                  cgMap.set(cg, groupId)
                  loadedMessages.push({ ...m, groupId })
                } else {
                  // Legacy - attach to current group (last user)
                  loadedMessages.push({ ...m, groupId: currentLegacyGroup || `group-${idx}` })
                }
              }
            })
            

            setMessages(loadedMessages)
            // Model/LLM selection handled by backendModels useEffect
            
            // Set model from messages or currentQuery
            if (backendModels.length > 0) {
              const modelIds = [...new Set(loadedMessages.filter(m => m.role === "assistant" && m.modelId).map(m => m.modelId))]
              if (modelIds.length > 0) {
                const models = modelIds.map(id => backendModels.find(m => m.id === id)).filter(Boolean) as typeof backendModels
                if (models.length > 0) {
                  setSelectedModels(models)
                  // Set source files from first model
                  if (models[0].datasets?.length > 0) {
                    setSelectedFiles(models[0].datasets.map(ds => ({
                      file_id: ds.datasetId,
                      filename: ds.datasetName,
                      source: ds.source
                    })))
                  }
                }
              } else if (currentQuery?.trainingModelId) {
                const model = backendModels.find(m => m.id === currentQuery.trainingModelId || m.name === currentQuery.trainingModelId)
                if (model) {
                  setSelectedModels([model])
                  if (model.datasets?.length > 0) {
                    setSelectedFiles(model.datasets.map(ds => ({
                      file_id: ds.datasetId,
                      filename: ds.datasetName,
                      source: ds.source
                    })))
                  }
                }
              }
            }
            
            setRefreshCount(c => c + 1)
          }
          setHasInitializedChat(true)
        })
        .catch(e => {
          console.error("Failed to load messages:", e)
          setHasInitializedChat(true)
        })
        .finally(() => setMessagesLoading(false))
    }
  }, [sessionId, hasInitializedChat])

  // Reset on new chat
  useEffect(() => {
    if (newChatTrigger && !sessionId) {
      setMessages([])
      setInput("")
      setCurrentQueryId(null)
      setHasInitializedChat(false)
      // Don't reset model if coming from build page with model param
      if (!modelIdFromUrl) {
        setSelectedModels([])
        setSelectedFiles([])
      }
    }
  }, [newChatTrigger, sessionId, modelIdFromUrl])

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isLoading])

  // Scroll button visibility
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const checkScroll = () => {
      const hasScroll = el.scrollHeight > el.clientHeight + 50
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
      setShowScrollButton(distanceFromBottom > 50)
    }
    el.addEventListener("scroll", checkScroll)
    checkScroll()
    return () => el.removeEventListener("scroll", checkScroll)
  }, [messages])

  const scrollToBottom = () => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }

  const toggleLLMSelection = (llmId: string) => {
    console.log("TOGGLE", llmId, "current:", selectedLLMs)
    const isSelected = selectedLLMs.includes(llmId)
    if (isSelected) {
      if (selectedLLMs.length > 1) {
        setSelectedLLMs(selectedLLMs.filter((id) => id !== llmId))
      }
    } else if (selectedLLMs.length < 2) {
      setSelectedLLMs([...selectedLLMs, llmId])
    }
  }

  const totalPages = Math.ceil(backendModels.length / MODELS_PER_PAGE)
  const paginatedModels = backendModels.slice(currentPage * MODELS_PER_PAGE, (currentPage + 1) * MODELS_PER_PAGE)

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as unknown as React.FormEvent)
    }
  }

  const primaryFile = selectedFiles[0]

  const buildDataContext = () => {
    if (selectedFiles.length === 0) return ""
    let context = ""
    selectedFiles.forEach((file: any) => {
      context += "- File: " + file.filename + "\n"
      if (file.size) context += "- Size: " + file.size + " bytes\n"
      if (file.columns) context += "- Columns: " + (Array.isArray(file.columns) ? file.columns.join(", ") : String(file.columns)) + "\n"
      if (file.unique_values) context += "- Target classes: " + file.unique_values.join(", ") + "\n"
      if (file.row_count) context += "- Rows: " + file.row_count + "\n"
    })
    return context
  }


  const toggleModelSelection = (model: AdaptedModel) => {
    if (compareMode) {
      const isSelected = selectedModels.some(m => m.id === model.id)
      if (isSelected) {
        if (selectedModels.length > 1) setSelectedModels(selectedModels.filter(m => m.id !== model.id))
      } else if (selectedModels.length < 4) {
        setSelectedModels([...selectedModels, model])
      }
    } else {
      setSelectedModels([model])
      const modelFiles = model.datasets && model.datasets.length > 0
        ? model.datasets.map(ds => ({ file_id: ds.datasetId, filename: ds.datasetName, source: ds.source }))
        : []
      setSelectedFiles(modelFiles)
    }
  }

  const toggleCompareMode = () => {
    if (!compareMode) {
      if (selectedModel && !selectedModels.some(m => m.id === selectedModel.id)) {
        setSelectedModels([selectedModel])
      }
      setSelectedLLMs([selectedLLMs[0] || "claude-sonnet-4-5"])
    } else {
      setSelectedModels(selectedModels.length > 0 ? [selectedModels[0]] : [])
    }
    setCompareMode(!compareMode)
  }

  // Restore compare mode from URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const compareModelIds = params.get("compare")
    if (compareModelIds && backendModels.length > 0) {
      const ids = compareModelIds.split(",")
      const models = ids.map(id => backendModels.find(m => m.id === id)).filter(Boolean) as AdaptedModel[]
      if (models.length > 1) {
        setCompareMode(true)
        setSelectedModels(models)
      }
    }
  }, [backendModels])

  // Save compare state to URL
  useEffect(() => {
    if (compareMode && selectedModels.length > 1) {
      const params = new URLSearchParams(window.location.search)
      params.set("compare", selectedModels.map(m => m.id).join(","))
      window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`)
    } else {
      const params = new URLSearchParams(window.location.search)
      if (params.has("compare")) {
        params.delete("compare")
        const qs = params.toString()
        window.history.replaceState({}, "", qs ? `${window.location.pathname}?${qs}` : window.location.pathname)
      }
    }
  }, [compareMode, selectedModels])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading || (selectedModels.length === 0 && !sessionId)) return

    const userMessage = input.trim()
    let queryId = currentQueryId || sessionId
    const startTime = Date.now()

    // Create new query if first message
    if (!queryId) {
      try {
        const createData = await api.createQuery(
          userMessage.substring(0, 50) || selectedModel?.name || "New Chat",
          selectedLLMs[0],
          [selectedModel?.id || ""],
          "",
          selectedModel?.name || "",
          selectedModel?.accuracy || 0,
          selectedModel?.datasets?.[0]?.datasetName || "",
          selectedModel?.id || ""
        )
        if (createData.id) {
          queryId = createData.id
          setCurrentQueryId(queryId)
          
          window.history.replaceState({}, '', `/playground/${queryId}`)
          
          // Add to sidebar
          setChatSessions(prev => prev.some(s => s.id === queryId) ? prev : [{
            id: queryId!,
            name: userMessage.substring(0, 50) || selectedModel?.name || "New Chat",
            modelIds: [selectedModel?.id || ""],
            llmIds: selectedLLMs,
            messages: [],
            createdAt: new Date(),
            updatedAt: new Date(),
          }, ...prev])
        }
      } catch (e) {
        console.error("Failed to create query:", e)
        return
      }
    }

    // Add user message
    const userMsg: DisplayMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: userMessage,
      timestamp: new Date(),
    }
    setMessages(prev => [...prev, userMsg])
    setInput("")
    setIsLoading(true)

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }

    const groupId = `group-${Date.now()}`

    // MODEL COMPARE: If compare mode with multiple models
    if (compareMode && selectedModels.length > 1) {
      const llmName = llmOptions.find(l => l.id === selectedLLMs[0])?.name || selectedLLMs[0]
      const assistantMsgs: DisplayMessage[] = selectedModels.map((model, idx) => ({
        id: `assistant-${Date.now()}-model-${idx}`,
        role: "assistant" as const,
        content: "",
        model: llmName,
        modelId: model.id,
        timestamp: new Date(),
        isLoading: true,
        groupId,
      }))
      setMessages(prev => [...prev, ...assistantMsgs])

      const llmId = selectedLLMs[0]
      const isClaudeModel = llmId.startsWith("claude")
      const compareGroupId = `cg-${Date.now()}`

      const promises = selectedModels.map(async (model, idx) => {
        let streamContent = ""
        const primaryFile = model.datasets?.[0]
        const modelFileId = primaryFile?.datasetId || ""
        const modelFileName = primaryFile?.datasetName || ""
        // Build model-specific data context
        const modelDataContext = model.datasets?.length > 0 
          ? model.datasets.map((ds: any) => `- File: ${ds.datasetName}`).join("\n") 
          : ""
        console.log(`COMPARE MODEL ${idx}: ${model.name}, fileId=${modelFileId}, fileName=${modelFileName}, dataContext=${modelDataContext}`)
        if (isClaudeModel) {
          const response = await api.chat({
            message: userMessage,
            file_id: modelFileId,
            query_id: queryId!,
            filename: modelFileName,
            model: llmId,
            data_context: modelDataContext,
            finetuned_model: model.id,
            model_path: model.modelPath || model.name,
            compare_group: compareGroupId,
          })
          const endTime = Date.now()
          const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
          if (response.error) { toast.error(response.error); return { modelId: model.id, content: "" + response.error, tokens: 0, time: "0s" } }
          return { modelId: model.id, content: response.response || "No response", tokens: response.tokens, time: timeTaken + "s", functionCalls: response.function_calls || [] }
        } else {
          return new Promise<{ modelId: string; content: string; tokens: number; time: string }>((resolve) => {
            api.chatStream(
              {
                message: userMessage,
                file_id: modelFileId,
                query_id: queryId!,
                filename: modelFileName,
                model: llmId,
                data_context: buildDataContext(),
                finetuned_model: model.id,
                model_path: model.modelPath || model.name,
                compare_group: compareGroupId || "",
              },
              (chunk) => {
                streamContent += chunk
                setMessages(prev => {
                  const newMessages = [...prev]
                  const msgIdx = newMessages.findIndex(m => m.modelId === model.id && m.groupId === groupId)
                  if (msgIdx !== -1) {
                    newMessages[msgIdx] = { ...newMessages[msgIdx], content: streamContent }
                  }
                  return newMessages
                })
              },
              () => {
                const endTime = Date.now()
                const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
                resolve({ modelId: model.id, content: streamContent, tokens: Math.round(streamContent.length / 4), time: timeTaken + "s" })
              }
            )
          })
        }
      })

      const results = await Promise.all(promises)
      if (response.error) {
          toast.error(response.error)
          setMessages(prev => {
            const newMessages = [...prev]
            const lastIdx = newMessages.length - 1
            newMessages[lastIdx] = { ...newMessages[lastIdx], content: "" + response.error, tokens: 0, time: "0s", isLoading: false }
            return newMessages
          })
          setIsLoading(false)
          return
        }
        setMessages(prev => {
        const newMessages = [...prev]
        results.forEach(result => {
          const msgIdx = newMessages.findIndex(m => m.modelId === result.modelId && m.groupId === groupId)
          if (msgIdx !== -1) {
            newMessages[msgIdx] = { ...newMessages[msgIdx], content: result.content, tokens: result.tokens, time: result.time, isLoading: false, functionCalls: result.functionCalls }
          }
        })
        return newMessages
      })
      setIsLoading(false)
      return
    }

    // If 2 LLMs selected, send to both
    if (selectedLLMs.length === 2) {
      const compareGroupId = `cg-${Date.now()}`
      // Add empty assistant messages for both LLMs
      const assistantMsgs: DisplayMessage[] = selectedLLMs.map((llmId, idx) => ({
        id: `assistant-${Date.now()}-${idx}`,
        role: "assistant" as const,
        content: "",
        model: llmOptions.find(l => l.id === llmId)?.name || llmId,
        llmId: llmId,
        timestamp: new Date(),
        isLoading: true,
        groupId,
      }))
      setMessages(prev => [...prev, ...assistantMsgs])
      
      // Send requests to both LLMs
      const promises = selectedLLMs.map(async (llmId, idx) => {
        const isClaudeModel = llmId.startsWith("claude")
        let streamContent = ""
        
        if (isClaudeModel) {
          const response = await api.chat({
            message: userMessage,
            file_id: selectedModel?.datasets?.[0]?.datasetId || primaryFile?.file_id || "",
            query_id: queryId!,
            filename: selectedModel?.datasets?.[0]?.datasetName || primaryFile?.filename || "",
            model: llmId,
            data_context: buildDataContext(),
            finetuned_model: selectedModel?.id || "",
            model_path: selectedModel?.modelPath || selectedModel?.name || "",
            compare_group: compareGroupId,
          })
          const endTime = Date.now()
          const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
          if (response.error) { toast.error(response.error); return { llmId, content: "" + response.error, tokens: 0, time: "0s" } }
          return { llmId, content: response.response || "No response", tokens: response.tokens, time: timeTaken + "s" }
        } else {
          return new Promise<{ llmId: string; content: string; tokens: number; time: string }>((resolve) => {
            api.chatStream(
              {
                message: userMessage,
                file_id: selectedModel?.datasets?.[0]?.datasetId || primaryFile?.file_id || "",
                query_id: queryId!,
                filename: selectedModel?.datasets?.[0]?.datasetName || primaryFile?.filename || "",
                model: llmId,
                data_context: buildDataContext(),
                finetuned_model: selectedModel?.id || "",
                model_path: selectedModel?.modelPath || selectedModel?.name || "",
                compare_group: compareGroupId,
              },
              (chunk) => {
                streamContent += chunk
                setMessages(prev => {
                  const newMessages = [...prev]
                  const msgIdx = newMessages.findIndex(m => m.llmId === llmId && m.groupId === groupId)
                  if (msgIdx !== -1) {
                    newMessages[msgIdx] = { ...newMessages[msgIdx], content: streamContent }
                  }
                  return newMessages
                })
              },
              () => {
                const endTime = Date.now()
                const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
                resolve({ llmId, content: streamContent, tokens: Math.round(streamContent.length / 4), time: timeTaken + "s" })
              }
            )
          })
        }
      })
      
      const results = await Promise.all(promises)
      setMessages(prev => {
        const newMessages = [...prev]
        results.forEach(result => {
          const msgIdx = newMessages.findIndex(m => m.llmId === result.llmId && m.groupId === groupId)
          if (msgIdx !== -1) {
            newMessages[msgIdx] = {
              ...newMessages[msgIdx],
              content: result.content,
              tokens: result.tokens,
              time: result.time,
              isLoading: false,
              functionCalls: result.functionCalls,
            }
          }
        })
        return newMessages
      })
      setIsLoading(false)
      return
    }

    // Single LLM flow
    const assistantMsgId = `assistant-${Date.now()}`
    const assistantMsg: DisplayMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      model: llmOptions.find(l => l.id === selectedLLMs[0])?.name || selectedLLMs[0],
      llmId: selectedLLMs[0],
      timestamp: new Date(),
      isLoading: true,
      groupId,
    }
    setMessages(prev => [...prev, assistantMsg])

    // Check if Claude model (non-streaming) or OpenAI (streaming)
    const isClaudeModel = selectedLLMs[0]?.startsWith("claude") || selectedLLMs[0]?.startsWith("gemini") || selectedLLMs[0]?.startsWith("mistral") || selectedLLMs[0]?.startsWith("ministral")
    let streamContent = ""

    try {
      if (isClaudeModel) {
        // Non-streaming for Claude
        const response = await api.chat({
          message: userMessage,
          file_id: selectedModel?.datasets?.[0]?.datasetId || primaryFile?.file_id || "",
          query_id: queryId!,
          filename: selectedModel?.datasets?.[0]?.datasetName || primaryFile?.filename || "",
          model: selectedLLMs[0],
          data_context: buildDataContext(),
          finetuned_model: selectedModel?.id || "",
          model_path: selectedModel?.modelPath || selectedModel?.name || "",
          compare_group: `sg-${Date.now()}`,
        })
        
        const endTime = Date.now()
        const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
        
        if (response.error) {
          const isKeyError = response.error.includes("API key") || response.error.includes("401") || response.error.includes("Unauthorized") || response.error.includes("api_key")
          if (isKeyError) {
            toast.error("API key required. Go to Configuration to add your key.", { duration: 5000 })
          } else {
            toast.error(response.error)
          }
          setMessages(prev => {
            const newMessages = [...prev]
            const lastIdx = newMessages.length - 1
            newMessages[lastIdx] = {
              ...newMessages[lastIdx],
              content: isKeyError ? "API key not found. Please go to Configuration to add your API key." : "" + response.error,
              tokens: 0,
              time: "0s",
              isLoading: false,
            }
            return newMessages
          })
          setIsLoading(false)
          return
        }
        
        setMessages(prev => {
          const newMessages = [...prev]
          const lastIdx = newMessages.length - 1
          newMessages[lastIdx] = {
            ...newMessages[lastIdx],
            content: response.response || "No response",
            tokens: response.tokens,
            time: timeTaken + "s",
            isLoading: false,
            functionCalls: response.function_calls || [],
          }
          return newMessages
        })
        setIsLoading(false)
        return
      } else {
        // Streaming for OpenAI
        await api.chatStream(
          {
            message: userMessage,
            file_id: selectedModel?.datasets?.[0]?.datasetId || primaryFile?.file_id || "",
            query_id: queryId!,
            filename: selectedModel?.datasets?.[0]?.datasetName || primaryFile?.filename || "",
            model: selectedLLMs[0],
            data_context: buildDataContext(),
            finetuned_model: selectedModel?.id || "",
            model_path: selectedModel?.modelPath || selectedModel?.name || "",
            compare_group: `sg-${Date.now()}`,
          },
          (chunk) => {
            streamContent += chunk
            setMessages(prev => {
              const newMessages = [...prev]
              const lastIdx = newMessages.length - 1
              newMessages[lastIdx] = {
                ...newMessages[lastIdx],
                content: streamContent,
              }
              return newMessages
            })
          },
          () => {
            const endTime = Date.now()
            const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
            setMessages(prev => {
              const newMessages = [...prev]
              const lastIdx = newMessages.length - 1
              newMessages[lastIdx] = {
                ...newMessages[lastIdx],
                time: timeTaken + "s",
                tokens: Math.round(streamContent.length / 4),
                isLoading: false,
              }
              return newMessages
            })
            // Silent refresh
          }
        )
      }
    } catch (e) {
      console.error("Chat error:", e)
      setMessages(prev => {
        const newMessages = [...prev]
        const lastIdx = newMessages.length - 1
        newMessages[lastIdx] = {
          ...newMessages[lastIdx],
          content: "Sorry, there was an error processing your request.",
          isLoading: false,
        }
        return newMessages
      })
    }

    setIsLoading(false)
  }

  const formatAccuracy = (acc: number) => {
    if (acc <= 0) return ""
    if (acc > 1) return `${acc.toFixed(2)}%`
    return `${(acc * 100).toFixed(2)}%`
  }

  // Loading state - only show if no sessionId
  if (modelsLoading && !sessionId) {
    return (
      <div className="flex h-[calc(100vh-6rem)] flex-col items-center justify-center px-4">
        <Loader2 className="h-8 w-8 animate-spin text-[#0052CC] dark:text-[#2684FF]" />
        <p className="mt-4 text-sm text-muted-foreground">Loading models...</p>
      </div>
    )
  }

  // Filter models by search query
  const filteredModels = modelSearchQuery
    ? backendModels.filter((m) =>
        m.name.toLowerCase().includes(modelSearchQuery.toLowerCase()) ||
        m.datasets.some((d) => d.datasetName.toLowerCase().includes(modelSearchQuery.toLowerCase()))
      )
    : backendModels

  // Model click handler for new chat screen
  const handleModelSelect = async (model: AdaptedModel) => {
    if (compareMode) {
      toggleModelSelection(model)
      return
    }
    setSelectedModels([model])
    setModelSearchQuery("")
    setIsLoading(true)
    const modelFiles = model.datasets && model.datasets.length > 0
      ? model.datasets.map(ds => ({
          file_id: ds.datasetId,
          filename: ds.datasetName,
          source: ds.source
        }))
      : []
    setSelectedFiles(modelFiles)
    
    try {
      if (!model.id) {
        console.error("Model ID is empty!")
        setMessages([])
        setHasInitializedChat(true)
        return
      }
      if (new URLSearchParams(window.location.search).get("new")) {
        setMessages([])
        setCurrentQueryId(null)
        setHasInitializedChat(true)
        setIsLoading(false)
        return
      }
      const msgData = await api.getMessages("", model.id)
      if (msgData.messages && Array.isArray(msgData.messages) && msgData.messages.length > 0) {
        const rawMessages = msgData.messages.map((m: any) => ({
          id: m.id,
          role: m.role as "user" | "assistant",
          content: m.content,
          model: m.model,
          tokens: m.tokens,
              time: m.time_taken || "",
          timestamp: new Date(m.created_at),
          query_id: m.query_id,
        }))
        
        const loadedMessages: DisplayMessage[] = []
        let currentGroupId: string | null = null
        
        rawMessages.forEach((m: any, idx: number) => {
          if (m.role === "user") {
            currentGroupId = `group-${m.id || idx}`
            loadedMessages.push({ ...m, groupId: currentGroupId })
          } else {
            loadedMessages.push({ ...m, groupId: currentGroupId || `group-${idx}` })
          }
        })
        setMessages(loadedMessages)
        
        const lastMsg = msgData.messages[msgData.messages.length - 1]
        if (lastMsg?.query_id) {
          setCurrentQueryId(lastMsg.query_id)
          window.history.pushState({}, "", `/playground/${lastMsg.query_id}`)
        }
      } else {
        setMessages([])
        setCurrentQueryId(null)
      }
    } catch (err) {
      console.error("Failed to load model messages:", err)
      setMessages([])
    }
    setHasInitializedChat(true)
    setIsLoading(false)
  }

  // No model selected - show model selection screen
  if (!selectedModel && !sessionId && !modelsLoading && !modelIdFromUrl) {
    return (
      <div className="flex h-[calc(100vh-4rem)] md:h-[calc(100vh-6rem)] flex-col">
        <div className="flex-1 flex items-center justify-center px-4">
          <div className="w-full max-w-2xl">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-foreground">New Chat</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Select a model to start querying your data
              </p>
            </div>

            {/* Search */}
            <div className="relative mb-4">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                value={modelSearchQuery}
                onChange={(e) => setModelSearchQuery(e.target.value)}
                placeholder="Search models..."
                className="pl-10 bg-card border-border"
              />
            </div>

            {backendModels.length === 0 ? (
              <div className="rounded-xl border border-border bg-card p-8 text-center">
                <div className="flex h-12 w-12 mx-auto items-center justify-center rounded-xl bg-muted">
                  <Box className="h-6 w-6 text-muted-foreground" />
                </div>
                <p className="mt-4 text-sm text-muted-foreground">No models built yet.</p>
                <Button
                  className="mt-4 bg-[#0052CC] hover:bg-[#003D99] text-white"
                  onClick={() => router.push("/build")}
                >
                  Build Your First Model
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {(modelSearchQuery ? filteredModels : paginatedModels).map((model) => (
                  <button
                    key={model.id}
                    onClick={() => handleModelSelect(model)}
                    className="flex flex-col rounded-xl border border-border bg-card p-4 text-left transition-all hover:border-[#0052CC]/40 hover:bg-[#0052CC]/5 dark:hover:border-[#2684FF]/40 dark:hover:bg-[#2684FF]/5"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#2684FF]/20">
                        <Box className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-medium text-foreground truncate">{model.name}</h4>
                        <p className="text-xs text-muted-foreground">
                          {formatAccuracy(model.accuracy)}
                        </p>
                      </div>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {model.datasets.slice(0, 2).map((ds) => (
                        <span key={ds.datasetId} className="inline-flex items-center gap-1 rounded-md bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                          <Database className="h-2.5 w-2.5" />
                          {ds.datasetName}
                        </span>
                      ))}
                      {model.datasets.length > 2 && (
                        <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                          +{model.datasets.length - 2} more
                        </span>
                      )}
                      {model.datasets.length === 0 && (
                        <span className="text-[10px] text-muted-foreground">{(model as any).connectionName || "No data sources"}</span>
                      )}
                    </div>
                  </button>
                ))}
                {filteredModels.length === 0 && (
                  <div className="col-span-2 py-8 text-center text-sm text-muted-foreground">
                    No models match your search
                  </div>
                )}
              </div>
            )}

            {modelSearchQuery === "" && totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 pt-4">
                <Button variant="outline" size="sm" onClick={() => setCurrentPage(p => Math.max(0, p - 1))} disabled={currentPage === 0} className="h-8 w-8 p-0">
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-sm text-muted-foreground">{currentPage + 1} / {totalPages}</span>
                <Button variant="outline" size="sm" onClick={() => setCurrentPage(p => Math.min(totalPages - 1, p + 1))} disabled={currentPage === totalPages - 1} className="h-8 w-8 p-0">
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
            <p className="text-center text-xs text-muted-foreground mt-4">{backendModels.length} model{backendModels.length !== 1 ? "s" : ""} available</p>
          </div>
        </div>
      </div>
    )
  }

  // Messages loading
  if (messagesLoading) {
    return (
      <div className="flex h-[calc(100vh-6rem)] flex-col items-center justify-center px-4">
        <Loader2 className="h-8 w-8 animate-spin text-[#0052CC] dark:text-[#2684FF]" />
        <p className="mt-4 text-sm text-muted-foreground">Loading conversation...</p>
      </div>
    )
  }

  if (!mounted) return null

  return (
    <>
    <TooltipProvider>
      <div className="flex h-[calc(100vh-1.5rem)] flex-col relative -mb-6">
        {/* Header Controls */}
        <div className="shrink-0 border-b border-border px-3 md:px-4 py-2 md:py-3">
          <div className="flex items-center justify-between gap-2 md:gap-4">
            {/* Left: Model selection + data sources */}
            <div className="flex items-center gap-3">
              <DropdownMenu modal={false}>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="gap-2 border-border bg-card">
                    <Box className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />
                    <span className="truncate max-w-[150px]">
                      {compareMode ? `${selectedModels.length} Model${selectedModels.length !== 1 ? "s" : ""}` : selectedModel?.name || "\u00A0"}
                    </span>
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-64">
                  <DropdownMenuLabel className="text-xs text-muted-foreground">
                    {compareMode ? "Select up to 4 models" : "Select a model"}
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {backendModels.map((model) => {
                    const isSelected = selectedModels.some((m) => m.id === model.id)
                    return (
                      <div key={model.id} onClick={() => toggleModelSelection(model)} className={cn("flex items-center gap-3 px-2 py-2 cursor-pointer rounded-md transition-colors", isSelected ? "bg-[#0052CC]/10 dark:bg-[#2684FF]/10" : "hover:bg-muted")}>
                        {compareMode && <div className={cn("flex h-4 w-4 shrink-0 items-center justify-center rounded border", isSelected ? "bg-[#0052CC] border-[#0052CC] dark:bg-[#2684FF] dark:border-[#2684FF]" : "border-muted-foreground/30")}>{isSelected && <Check className="h-3 w-3 text-white" />}</div>}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{model.name}</p>
                          <p className="text-xs text-muted-foreground">{model.datasets.length > 0 ? `${model.datasets.length} source${model.datasets.length !== 1 ? "s" : ""}` : (model as any).isConnectionBased ? ((model as any).connectionName || "connection") : "no sources"}</p>
                        </div>
                        {!compareMode && isSelected && <Check className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />}
                      </div>
                    )
                  })}
                </DropdownMenuContent>
              </DropdownMenu>

              {compareMode && selectedModels.length > 1 && (
                <div className="flex items-center gap-1.5 flex-wrap">
                  {selectedModels.slice(1).map((model) => (
                    <div key={model.id} className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs bg-muted text-muted-foreground">
                      {model.name}
                      <button onClick={() => setSelectedModels(selectedModels.filter((m) => m.id !== model.id))} className="hover:opacity-70">×</button>
                    </div>
                  ))}
                </div>
              )}

              {/* Connected data tooltip */}
              {(() => {
                const allSources = compareMode && selectedModels.length > 1
                  ? selectedModels.flatMap(m => m.datasets || [])
                  : (selectedModel?.datasets || [])
                return allSources.length > 0 ? (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
                      <Database className="h-3.5 w-3.5" />
                      <span>{allSources.length} source{allSources.length !== 1 ? "s" : ""}</span>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="max-w-xs p-3 bg-popover border border-border">
                    <p className="text-xs font-medium mb-2 text-foreground">Connected Data Sources</p>
                    <div className="space-y-1.5">
                      {allSources.map((ds: any) => (
                        <div key={ds.datasetId} className="flex items-center gap-2 text-xs">
                          <Database className="h-3 w-3 text-muted-foreground" />
                          <span className="text-foreground">{ds.datasetName}</span>
                        </div>
                      ))}
                    </div>
                  </TooltipContent>
                </Tooltip>
              ) : null})()}
            </div>

            {/* Right: Compare mode */}
            <div className="flex items-center gap-2">
              {selectedLLMs.length === 2 && !compareMode && (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#0052CC]/10 text-xs text-[#0052CC] dark:text-[#2684FF]">
                  <GitCompare className="h-3.5 w-3.5" />
                  <span>Compare ({selectedLLMs.length} LLMs)</span>
                </div>
              )}
              <Button
                variant={compareMode ? "default" : "outline"}
                size="sm"
                onClick={toggleCompareMode}
                className={cn("gap-1.5", compareMode && "bg-[#0052CC] hover:bg-[#003D99] text-white")}
              >
                <GitCompare className="h-3.5 w-3.5" />
                Compare
                {compareMode && ` (${selectedModels.length})`}
              </Button>
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden relative" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center px-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted">
                <Box className="h-6 w-6 text-muted-foreground" />
              </div>
              <p className="mt-4 text-sm text-muted-foreground">Start chatting with your data</p>
            </div>
          ) : (
            <div key={refreshCount} className={cn("mx-auto p-4 space-y-6", compareMode ? "max-w-full px-4" : "max-w-4xl")}>
              {(() => {
                const renderedGroups = new Set<string>()
                return messages.map((msg, idx) => {
                  // Skip if this is part of a group we already rendered
                  if (msg.groupId && msg.role === "assistant" && renderedGroups.has(msg.groupId)) {
                    return null
                  }
                  
                  // Check if this is a compare group (multiple assistant messages with same groupId)
                  const groupMessages = msg.groupId ? messages.filter(m => m.groupId === msg.groupId && m.role === "assistant") : []
                  const isCompareGroup = groupMessages.length > 1
                  
                  if (isCompareGroup && msg.role === "assistant") {
                    renderedGroups.add(msg.groupId!)
                    return (
                      <div key={msg.groupId} className={cn("grid gap-3 grid-cols-1", groupMessages.length === 2 && "sm:grid-cols-2", groupMessages.length === 3 && "sm:grid-cols-2 lg:grid-cols-3", groupMessages.length >= 4 && "sm:grid-cols-2 xl:grid-cols-4")}>
                        {groupMessages.map((compareMsg) => (
                          <div key={compareMsg.id} className="space-y-1 min-w-0">
                            <span className="text-xs font-medium px-1 text-muted-foreground">
                              {(compareMsg.modelId && backendModels.find(m => m.id === compareMsg.modelId)?.name) || selectedModel?.name || compareMsg.model}
                            </span>
                            <div className="rounded-2xl rounded-tl-md border border-border bg-card p-4 overflow-hidden break-words max-w-full">
                              {compareMsg.isLoading && !compareMsg.content ? (
                                <div className="flex gap-1">
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.2s]" />
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.4s]" />
                                </div>
                              ) : (
                                <>
                                {compareMsg.functionCalls && compareMsg.functionCalls.length > 0 && (
                                  <FunctionCallDisplay calls={compareMsg.functionCalls} />
                                )}
                                <ContentRenderer content={compareMsg.content} />
                                </>
                              )}
                              {(compareMsg.tokens || compareMsg.time) && !compareMsg.isLoading && (
                                <div className="flex items-center gap-3 mt-3 pt-3 border-t border-border">
                                  {compareMsg.tokens && (
                                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                      <Zap className="h-3 w-3" /> {compareMsg.tokens} tokens
                                    </span>
                                  )}
                                  {compareMsg.time && (
                                    <span className="text-xs text-muted-foreground">{compareMsg.time}</span>
                                  )}
                                  <span className="text-[10px] text-muted-foreground/70 ml-auto">
                                    {compareMsg.model}
                                  </span>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )
                  }
                  
                  return (
                    <div key={msg.id} className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}>
                      {msg.role === "user" ? (
                        <div className="flex items-start gap-3 max-w-[80%]">
                          <div className="rounded-2xl rounded-tr-md bg-[#0052CC] px-4 py-2.5 text-white">
                            <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                          </div>
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                            <User className="h-4 w-4 text-muted-foreground" />
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-start gap-3">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                            <Box className="h-4 w-4 text-muted-foreground" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="rounded-2xl rounded-tl-md border border-border bg-card p-4 overflow-hidden break-words max-w-full">

                              {msg.isLoading && !msg.content ? (
                                <div className="flex gap-1">
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.2s]" />
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.4s]" />
                                </div>
                              ) : (
                                <>
                                {msg.functionCalls && msg.functionCalls.length > 0 && (
                                  <FunctionCallDisplay calls={msg.functionCalls} />
                                )}
                                <ContentRenderer content={msg.content} />
                                </>
                              )}
                              {(msg.tokens || msg.time) && !msg.isLoading && (
                                <div className="flex items-center gap-3 mt-3 pt-3 border-t border-border">
                                  {msg.tokens && (
                                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                      <Zap className="h-3 w-3" /> {msg.tokens} tokens
                                    </span>
                                  )}
                                  {msg.time && (
                                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                      <Clock className="h-3 w-3" /> {msg.time}
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })
              })()}
            </div>
          )}
        </div>

        {/* Scroll to bottom button */}
        {showScrollButton && (
          <button
            onClick={scrollToBottom}
            className="absolute bottom-44 left-1/2 -translate-x-1/2 p-2 bg-background border border-border rounded-full shadow-lg hover:bg-accent transition-all z-10"
          >
            <ArrowDown className="h-4 w-4" />
          </button>
        )}

        {/* Input Area */}
        <div className="shrink-0 px-4 pb-0 pt-2">
          <form onSubmit={handleSubmit} className={cn("mx-auto", compareMode ? "max-w-full px-4" : "max-w-4xl")}>
            <div className="rounded-2xl border border-border p-3">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder={selectedModel ? "Ask anything about your data..." : "Select a model first..."}
                className="min-h-[44px] max-h-[200px] resize-none border-0 bg-transparent p-0 text-sm focus-visible:ring-0 placeholder:text-muted-foreground"
                disabled={isLoading || (!selectedModel && !sessionId)}
                rows={1}
              />
              <div className="mt-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button type="button" variant="ghost" size="sm" className="h-8 w-8 p-0 text-muted-foreground hover:text-[#0052CC]" onClick={() => setVerticalPanelOpen(true)}>
                        <Plus className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      <p>Vertical AI Runtime</p>
                    </TooltipContent>
                  </Tooltip>

                <DropdownMenu open={llmDropdownOpen} onOpenChange={setLlmDropdownOpen}>
                  <DropdownMenuTrigger asChild>
                    <Button type="button" variant="ghost" size="sm" className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground">
                      <Settings2 className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="w-72">
                    <DropdownMenuLabel className="text-xs text-muted-foreground uppercase tracking-wider">LLM Provider</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {llmOptions.filter((llm) => llm.provider === "Schema" || keyStatus.unlimited || keyStatus[llm.provider === "Google" ? "gemini" : llm.provider.toLowerCase()]).map((llm) => {
                      const isSelected = selectedLLMs.includes(llm.id)
                      return (
                        <div
                          key={llm.id}
                          onClick={() => {
                            const hasKey = true
                            if (!hasKey) return
                            toggleLLMSelection(llm.id)
                          }}
                          className={cn("flex items-center gap-3 px-2 py-2 rounded-md transition-colors", isSelected ? "bg-[#0052CC]/10 dark:bg-[#2684FF]/10 cursor-pointer" : "hover:bg-muted cursor-pointer")}
                        >
                          <Checkbox checked={isSelected} className="pointer-events-none" />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium">{llm.name}</p>
                            <p className="text-[10px] text-muted-foreground">{llm.provider}</p>
                          </div>
                          {!keyStatus.unlimited && llm.provider !== "Schema" && (
                            keyStatus[llm.provider === "Google" ? "gemini" : llm.provider.toLowerCase()] ? (
                              <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-500 shrink-0">Key</span>
                            ) : (
                              <button onClick={(e) => { e.stopPropagation(); const pKey = llm.provider === "Google" ? "gemini" : llm.provider.toLowerCase(); setAddKeyModal({open: true, provider: pKey, providerLabel: llm.provider}); setAddKeyValue(""); setLlmDropdownOpen(false) }} className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20 shrink-0">Add Key</button>
                            )
                          )}
                          {compareMode && isSelected && <Check className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />}
                        </div>
                      )
                    })}
                      <DropdownMenuSeparator />
                      <div className="px-2 py-1.5">
                        <p className="text-[10px] text-muted-foreground">
                          {compareMode ? "Single LLM in compare mode" : "Select up to 2 LLMs"}
                        </p>
                        {!keyStatus.unlimited && <button onClick={() => { setAddKeyModal({open: true, provider: "", providerLabel: ""}); setAddKeyValue(""); setLlmDropdownOpen(false) }} className="w-full mt-1 py-2 text-sm text-center rounded-md border border-dashed border-muted-foreground/30 text-muted-foreground hover:border-[#0052CC] hover:text-[#0052CC] dark:hover:border-[#2684FF] dark:hover:text-[#2684FF] transition-colors">+ Add LLM Provider</button>}
                      </div>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                {/* Add Key Modal */}
                {addKeyModal.open && (
                  <div className="fixed inset-0 z-50 flex items-center justify-center">
                    <div className="fixed inset-0 bg-black/50" onClick={() => setAddKeyModal({open: false, provider: "", providerLabel: ""})} />
                    <div className="relative z-50 w-[400px] rounded-lg border border-border bg-card p-6 shadow-xl">
                      <h3 className="text-sm font-semibold mb-1">Add LLM Provider</h3>
                      <p className="text-xs text-muted-foreground mb-4">Select a provider and enter your API key.</p>
                      <div className="flex gap-2 mb-4">
                        {[{key: "openai", label: "OpenAI"}, {key: "anthropic", label: "Anthropic"}, {key: "gemini", label: "Google"}].map(p => (
                          <button key={p.key} onClick={() => setAddKeyModal(prev => ({...prev, provider: p.key, providerLabel: p.label}))} className={`flex-1 py-2 text-xs font-medium rounded-md border transition-colors ${addKeyModal.provider === p.key ? "border-[#0052CC] bg-[#0052CC]/10 text-[#0052CC] dark:border-[#2684FF] dark:bg-[#2684FF]/10 dark:text-[#2684FF]" : "border-border hover:bg-muted"}`}>{p.label}</button>
                        ))}
                      </div>
                      <input type="password" placeholder={addKeyModal.provider ? `Enter ${addKeyModal.providerLabel} API key...` : "Select a provider first..."} disabled={!addKeyModal.provider} value={addKeyValue} onChange={e => setAddKeyValue(e.target.value)} className="w-full h-9 rounded-md border border-border bg-background px-3 text-sm font-mono mb-4 focus:outline-none focus:ring-1 focus:ring-[#2684FF] disabled:opacity-50" autoFocus />
                      <div className="flex justify-end gap-2">
                        <button onClick={() => setAddKeyModal({open: false, provider: "", providerLabel: ""})} className="px-3 py-1.5 text-sm rounded-md border border-border hover:bg-muted">Cancel</button>
                        <button disabled={!addKeyValue || !addKeyModal.provider || addKeySaving} onClick={async () => {
                          setAddKeySaving(true)
                          try {
                            const testRes = await fetch("/api/vertical/secrets/test", { method: "POST", headers: {"Content-Type": "application/json"}, credentials: "include", body: JSON.stringify({provider: addKeyModal.provider, api_key: addKeyValue}) })
                            const testData = await testRes.json()
                            if (!testData.success) {
                              toast.error("Invalid API key: " + (testData.error || "Authentication failed"))
                              setAddKeySaving(false)
                              return
                            }
                            await fetch("/api/vertical/secrets", { method: "POST", headers: {"Content-Type": "application/json"}, credentials: "include", body: JSON.stringify({provider: addKeyModal.provider, secret_name: addKeyModal.provider.toUpperCase() + "_API_KEY", value: addKeyValue, vertical_id: ""}) })
                            setKeyStatus(prev => ({...prev, [addKeyModal.provider]: true}))
                            setAddKeyModal({open: false, provider: "", providerLabel: ""})
                            setAddKeyValue("")
                            toast.success("API key saved and verified")
                          } catch { toast.error("Failed to save") }
                          setAddKeySaving(false)
                        }} className="px-3 py-1.5 text-sm rounded-md bg-[#0052CC] text-white hover:bg-[#003D99] disabled:opacity-50">
                          {addKeySaving ? "Verifying..." : "Save Key"}
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                <Button
                  type="submit"
                  size="sm"
                  data-send-button
                  disabled={!input.trim() || isLoading || (!selectedModel && !sessionId)}
                  className={cn("h-8 w-8 rounded-full p-0 transition-colors", input.trim() ? "bg-[#0052CC] hover:bg-[#003D99] text-white" : "bg-muted text-muted-foreground")}
                >
                  <ArrowUp className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </form>
          <p className="text-center text-[11px] text-muted-foreground mt-1 mb-0 pb-1">v.Alpha: Outputs may be incorrect, verify important information.</p>
        </div>
      </div>
    </TooltipProvider>
    <VerticalPanel
      open={verticalPanelOpen}
      onClose={() => setVerticalPanelOpen(false)}
      modelId={selectedModel?.id || ""}
      modelName={selectedModel?.name || "No model selected"}
    />
    </>
  )
}
