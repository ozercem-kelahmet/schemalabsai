"use client"

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
  ArrowUp,
  ArrowDown,
  Box,
  Zap,
  Check,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { ContentRenderer, type ResponseBlock } from "./response-renderer"
import { api } from "@/lib/api"

const llmOptions = [
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", provider: "Anthropic" },
  { id: "claude-opus-4", name: "Claude Opus 4", provider: "Anthropic" },
  { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "OpenAI" },
]

interface BackendModel {
  id: string
  name: string
  accuracy?: number
  sourceCsvName?: string
  sourceFiles?: string
  source_files?: string
  source_csv_name?: string
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
  tokens?: number
  time?: string
  timestamp: Date
  isLoading?: boolean
}

function adaptBackendModel(m: BackendModel): AdaptedModel {
  console.log("DEBUG adaptBackendModel input:", m.name, "source_files:", m.source_files, "sourceFiles:", m.sourceFiles)
  const sourceFilesStr = m.sourceFiles || m.source_files || ""
  const sourceCsvName = m.sourceCsvName || m.source_csv_name || ""
  const sourceFiles = sourceFilesStr ? sourceFilesStr.split(",").filter(Boolean) : []
  const datasets = sourceFiles.length > 0 
    ? sourceFiles.map((file, idx) => ({
        datasetId: `ds-${m.id}-${idx}`,
        datasetName: file.trim(),
        source: "upload" as DataSource,
      }))
    : sourceCsvName 
    ? [{ datasetId: `ds-${m.id}`, datasetName: sourceCsvName, source: "upload" as DataSource }]
    : []

  return {
    id: m.id,
    name: m.name || "Unnamed Model",
    accuracy: m.accuracy || 0,
    datasets,
    sourceFiles: sourceFilesStr,
    modelPath: m.model_path || m.modelPath || "",
  }
}

const MODELS_PER_PAGE = 4

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
  const { chatSessions, setChatSessions } = useSidebar()
  const { queries, getQuery } = useQueryStore()
  const currentQuery = useMemo(() => {
    const q = sessionId ? getQuery(sessionId) : null
    console.log("DEBUG currentQuery:", q, "trainingModelId:", q?.trainingModelId)
    return q
  }, [sessionId, getQuery, queries])

  // Models state
  const [backendModels, setBackendModels] = useState<AdaptedModel[]>([])
  const [modelsLoading, setModelsLoading] = useState(true)
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [selectedFiles, setSelectedFiles] = useState<any[]>([])
  const [currentPage, setCurrentPage] = useState(0)
  const [selectedModel, setSelectedModel] = useState<AdaptedModel | null>(null)

  // Chat state
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [currentQueryId, setCurrentQueryId] = useState<string | null>(null)
  const [selectedLLMs, setSelectedLLMs] = useState<string[]>(["claude-sonnet-4-5"])

  // Silent refresh state
  const [hasInitializedChat, setHasInitializedChat] = useState(false)
  const [refreshCount, setRefreshCount] = useState(0)

  // Scroll state
  const [showScrollButton, setShowScrollButton] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  // UI state
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const [llmDropdownOpen, setLlmDropdownOpen] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Fetch models from backend
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await api.getFineTunedModels()
        console.log("DEBUG backend models raw:", data.models?.map((m: any) => ({ id: m.id, name: m.name })))
        if (data.models && Array.isArray(data.models)) {
          const adapted = data.models.map(adaptBackendModel)
          console.log("DEBUG adapted models:", adapted.map(m => ({ id: m.id, name: m.name })))
          setBackendModels(adapted)
        }
      } catch (e) {
        console.error("Failed to fetch models:", e)
      }
      setModelsLoading(false)
    }
    fetchModels()
  }, [])

  // Load uploaded files
  useEffect(() => {
    const loadFiles = async () => {
      try {
        const data = await api.getUploadedFiles()
        if (data.files) {
          setUploadedFiles(data.files)
        }
      } catch (e) {
        console.error("Failed to load files:", e)
      }
    }
    loadFiles()
  }, [])

  // Select model from URL parameter
  useEffect(() => {
    if (!modelIdFromUrl || backendModels.length === 0) return
    const model = backendModels.find(m => m.id === modelIdFromUrl)
    if (model) {
      setHasInitializedChat(true)
      setSelectedModel(model)
      if (model.datasets && model.datasets.length > 0) {
        setSelectedFiles(model.datasets.map(ds => ({
          file_id: ds.datasetId,
          filename: ds.datasetName,
          source: ds.source
        })))
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
        setSelectedModel(model)
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
        setSelectedModel(model)
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
        if (model) setSelectedModel(model)
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
              tokens: m.tokens,
              timestamp: new Date(m.created_at),
            }))
            
            // Assign groupIds - all assistants after a user get same groupId until next user
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
            const assistantMsg = loadedMessages.find((m: DisplayMessage) => m.role === "assistant" && m.model)
            if (assistantMsg?.model) {
              const modelLower = assistantMsg.model.toLowerCase()
              if (modelLower.includes("gpt")) setSelectedLLMs(["gpt-4o"])
              else if (modelLower.includes("claude")) setSelectedLLMs(["claude-sonnet-4-5"])
            }
            
            // Set model from currentQuery if available
            if (currentQuery?.trainingModelId && backendModels.length > 0) {
              const model = backendModels.find(m => m.id === currentQuery.trainingModelId || m.name === currentQuery.trainingModelId)
              console.log("DEBUG model from currentQuery:", model?.name)
              if (model) setSelectedModel(model)
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
      setSelectedModel(null)
      setSelectedFiles([])
      setMessages([])
      setInput("")
      setCurrentQueryId(null)
      setHasInitializedChat(false)
    }
  }, [newChatTrigger, sessionId])

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
    console.log("DEBUG buildDataContext - selectedFiles:", selectedFiles, "length:", selectedFiles.length)
    if (selectedFiles.length === 0) return ""
    let context = ""
    selectedFiles.forEach((file: any) => {
      context += "- File: " + file.filename + "\n"
      if (file.size) context += "- Size: " + file.size + " bytes\n"
      if (file.columns) context += "- Columns: " + file.columns.join(", ") + "\n"
      if (file.unique_values) context += "- Target classes: " + file.unique_values.join(", ") + "\n"
      if (file.row_count) context += "- Rows: " + file.row_count + "\n"
    })
    return context
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading || (!selectedModel && !sessionId)) return

    const userMessage = input.trim()
    let queryId = currentQueryId
    const startTime = Date.now()

    // Create new query if first message
    if (!queryId) {
      try {
        const createData = await api.createQuery(
          message.trim().substring(0, 50) || selectedModel.name,
          selectedLLMs[0],
          [selectedModel.id],
          "",
          selectedModel.name,
          selectedModel.accuracy,
          selectedModel.datasets[0]?.datasetName || "",
          selectedModel.id
        )
        if (createData.id) {
          queryId = createData.id
          setCurrentQueryId(queryId)
          
          window.history.pushState({}, "", `/playground/${queryId}`)
          
          // Add to sidebar
          setChatSessions(prev => [{
            id: queryId!,
            name: selectedModel?.name || "New Chat",
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
    
    // If 2 LLMs selected, send to both
    if (selectedLLMs.length === 2) {
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
            file_id: currentQuery?.dataSources?.[0] || primaryFile?.file_id || "",
            query_id: queryId!,
            filename: currentQuery?.sourceCsvName || selectedModel?.datasets?.[0]?.datasetName || "",
            model: llmId,
            data_context: buildDataContext(),
            finetuned_model: currentQuery?.trainingModelId || selectedModel?.id || "",
            model_path: selectedModel?.modelPath || selectedModel?.name || "",
          })
          const endTime = Date.now()
          const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
          return { llmId, content: response.response || "No response", tokens: response.tokens, time: timeTaken + "s" }
        } else {
          return new Promise<{ llmId: string; content: string; tokens: number; time: string }>((resolve) => {
            api.chatStream(
              {
                message: userMessage,
                file_id: currentQuery?.dataSources?.[0] || primaryFile?.file_id || "",
                query_id: queryId!,
                filename: currentQuery?.sourceCsvName || selectedModel?.datasets?.[0]?.datasetName || "",
                model: llmId,
                data_context: buildDataContext(),
                finetuned_model: currentQuery?.trainingModelId || selectedModel?.id || "",
                model_path: selectedModel?.modelPath || selectedModel?.name || "",
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
            }
          }
        })
        return newMessages
      })
      setIsLoading(false)
      setTimeout(() => setHasInitializedChat(false), 100)
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
    const isClaudeModel = selectedLLMs[0]?.startsWith("claude")
    let streamContent = ""

    try {
      if (isClaudeModel) {
        // Non-streaming for Claude
        const response = await api.chat({
          message: userMessage,
          file_id: currentQuery?.dataSources?.[0] || primaryFile?.file_id || "",
          query_id: queryId!,
          filename: currentQuery?.sourceCsvName || selectedModel?.datasets?.[0]?.datasetName || "",
          model: selectedLLMs[0],
          data_context: buildDataContext(),
          finetuned_model: currentQuery?.trainingModelId || selectedModel?.id || "",
          model_path: selectedModel?.modelPath || selectedModel?.name || "",
        })
        
        const endTime = Date.now()
        const timeTaken = ((endTime - startTime) / 1000).toFixed(1)
        
        setMessages(prev => {
          const newMessages = [...prev]
          const lastIdx = newMessages.length - 1
          newMessages[lastIdx] = {
            ...newMessages[lastIdx],
            content: response.response || "No response",
            tokens: response.tokens,
            time: timeTaken + "s",
            isLoading: false,
          }
          return newMessages
        })
        setIsLoading(false)
        setTimeout(() => setHasInitializedChat(false), 100)
        return
      } else {
        // Streaming for OpenAI
        await api.chatStream(
          {
            message: userMessage,
            file_id: currentQuery?.dataSources?.[0] || primaryFile?.file_id || "",
            query_id: queryId!,
            filename: currentQuery?.sourceCsvName || selectedModel?.datasets?.[0]?.datasetName || "",
            model: selectedLLMs[0],
            data_context: buildDataContext(),
            finetuned_model: currentQuery?.trainingModelId || selectedModel?.id || "",
            model_path: selectedModel?.modelPath || selectedModel?.name || "",
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
            setTimeout(() => setHasInitializedChat(false), 100)
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

  // No model selected - show model selection screen
  if (!selectedModel && !sessionId && !modelsLoading) {
    return (
      <div className="flex h-[calc(100vh-6rem)] flex-col items-center justify-center px-4">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#0052CC]/20 to-[#003D99]/20">
          <Box className="h-8 w-8 text-[#0052CC] dark:text-[#2684FF]" />
        </div>
        <h3 className="mt-4 text-lg font-medium text-foreground">Select a Model to Start</h3>
        <p className="mt-2 max-w-md text-center text-sm text-muted-foreground">
          Choose one of your built models to start chatting with your data
        </p>

        <div className="mt-8 w-full max-w-md space-y-3">
          {backendModels.length === 0 ? (
            <div className="text-center">
              <p className="text-sm text-muted-foreground">No models built yet.</p>
              <Button className="mt-4 bg-[#0052CC] hover:bg-[#003D99] text-white" onClick={() => router.push("/build")}>
                Build Your First Model
              </Button>
            </div>
          ) : (
            <>
              {paginatedModels.map((model) => (
                <button
                  key={model.id}
                  onClick={async (e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    
                    // Set the model immediately for UI feedback
                    setSelectedModel(model)
                    setIsLoading(true)
                    const modelFiles = model.datasets && model.datasets.length > 0
                      ? model.datasets.map(ds => ({
                          file_id: ds.datasetId,
                          filename: ds.datasetName,
                          source: ds.source
                        }))
                      : []
                    setSelectedFiles(modelFiles)
                    
                    // Load messages for this model directly (single API call)
                    try {
                      if (!model.id) {
                        console.error("Model ID is empty!")
                        setMessages([])
                        setHasInitializedChat(true)
                        return
                      }
                      // If new chat trigger, don't load old messages
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
                          timestamp: new Date(m.created_at),
                          query_id: m.query_id,
                        }))
                        
                        // Assign groupIds - all assistants after a user get same groupId until next user
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
                        
                        // Set query ID from last message
                        const lastMsg = msgData.messages[msgData.messages.length - 1]
                        if (lastMsg?.query_id) {
                          setCurrentQueryId(lastMsg.query_id)
                          window.history.pushState({}, '', `/playground/${lastMsg.query_id}`)
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
                  }}
                  className="w-full flex items-center gap-4 rounded-xl border border-border bg-card p-4 text-left transition-all hover:border-[#0052CC]/30 hover:bg-[#0052CC]/5"
                >
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/10">
                    <Box className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-foreground truncate">{model.name}</h4>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {model.datasets.length > 0 ? `${model.datasets.length} data source${model.datasets.length !== 1 ? "s" : ""}` : "No data sources"}
                    </p>
                  </div>
                  <div className="text-xs text-muted-foreground">{formatAccuracy(model.accuracy)}</div>
                </button>
              ))}

              {totalPages > 1 && (
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
              <p className="text-center text-xs text-muted-foreground pt-2">{backendModels.length} model{backendModels.length !== 1 ? "s" : ""} available</p>
            </>
          )}
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

  return (
    <TooltipProvider>
      <div className="flex h-[calc(100vh-6rem)] flex-col relative">
        {/* Header */}
        <div className="shrink-0 border-b border-border px-4 py-3">
          <div className="flex items-center gap-3">
            {/* Model Name */}
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-card hover:bg-muted transition-colors">
                  <Box className="h-4 w-4 text-[#0052CC]" />
                  <span className="text-sm font-medium truncate max-w-[200px]">{selectedModel?.name || currentQuery?.name || "Model"}</span>
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                </button>
              </TooltipTrigger>
              {selectedModel && (
                <TooltipContent side="bottom" className="max-w-xs p-3">
                  <p className="text-xs font-medium">{selectedModel.name}</p>
                  <p className="text-xs text-muted-foreground mt-1">Accuracy: {selectedModel.accuracy > 1 ? selectedModel.accuracy.toFixed(2) : (selectedModel.accuracy * 100).toFixed(2)}%</p>
                </TooltipContent>
              )}
            </Tooltip>

            {/* Source Badge */}
            {selectedFiles.length > 0 && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-border bg-card text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors">
                    <Database className="h-3.5 w-3.5" />
                    <span>{selectedFiles.length} source{selectedFiles.length !== 1 ? "s" : ""}</span>
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-md p-3 bg-popover border border-border">
                  <p className="text-xs font-medium mb-2 text-foreground">Connected Data Sources</p>
                  <div className="space-y-1.5">
                    {selectedFiles.map((file: any) => (
                      <div key={file.file_id} className="flex items-center gap-2 text-xs">
                        <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shrink-0">
                          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                          <span className="text-[10px] font-medium">File</span>
                        </div>
                        <span className="text-foreground">{file.filename}</span>
                      </div>
                    ))}
                  </div>
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto relative" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center px-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted">
                <Box className="h-6 w-6 text-muted-foreground" />
              </div>
              <p className="mt-4 text-sm text-muted-foreground">Start chatting with your data</p>
            </div>
          ) : (
            <div key={refreshCount} className="mx-auto max-w-4xl p-4 space-y-6">
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
                      <div key={msg.groupId} className="grid grid-cols-2 gap-4">
                        {groupMessages.map((compareMsg) => (
                          <div key={compareMsg.id} className="flex flex-col">
                            <div className="flex items-center gap-2 mb-2">
                              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-muted">
                                <Box className="h-3 w-3 text-muted-foreground" />
                              </div>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">{compareMsg.model}</span>
                            </div>
                            <div className="rounded-2xl border border-border bg-card p-4 flex-1">
                              {compareMsg.isLoading && !compareMsg.content ? (
                                <div className="flex gap-1">
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.2s]" />
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.4s]" />
                                </div>
                              ) : (
                                <ContentRenderer content={compareMsg.content} />
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
                        <div className="flex items-start gap-3 max-w-[90%]">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                            <Box className="h-4 w-4 text-muted-foreground" />
                          </div>
                          <div className="flex-1 space-y-2">
                            {msg.model && (
                              <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground mb-2 inline-block">{msg.model}</span>
                            )}
                            <div className="rounded-2xl rounded-tl-md border border-border bg-card p-4">
                              {msg.isLoading && !msg.content ? (
                                <div className="flex gap-1">
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.2s]" />
                                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.4s]" />
                                </div>
                              ) : (
                                <ContentRenderer content={msg.content} />
                              )}
                              {(msg.tokens || msg.time) && !msg.isLoading && (
                                <div className="flex items-center gap-3 mt-3 pt-3 border-t border-border">
                                  {msg.tokens && (
                                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                      <Zap className="h-3 w-3" /> {msg.tokens} tokens
                                    </span>
                                  )}
                                  {msg.time && (
                                    <span className="text-xs text-muted-foreground">{msg.time}</span>
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
        <div className="shrink-0 border-t border-border p-4">
          <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
            <div className="rounded-2xl border border-border bg-card p-3">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder={selectedModel ? "Ask anything about your data..." : "Select a model first..."}
                className="min-h-[44px] max-h-[200px] resize-none border-0 bg-transparent p-0 text-sm focus-visible:ring-0"
                disabled={isLoading || (!selectedModel && !sessionId)}
                rows={1}
              />
              <div className="mt-2 flex items-center justify-between">
                <DropdownMenu open={llmDropdownOpen} onOpenChange={setLlmDropdownOpen}>
                  <DropdownMenuTrigger asChild>
                    <Button type="button" variant="ghost" size="sm" className="h-8 gap-1.5 px-2 text-muted-foreground">
                      <span className="text-xs">{selectedLLMs.length === 1 ? llmOptions.find(l => l.id === selectedLLMs[0])?.name : `${selectedLLMs.length} LLMs`}</span>
                      <ChevronDown className="h-3 w-3" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="w-56">
                    <DropdownMenuLabel className="text-xs text-muted-foreground">Select up to 2 LLMs</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {llmOptions.map((llm) => {
                      const isSelected = selectedLLMs.includes(llm.id)
                      return (
                        <div
                          key={llm.id}
                          onClick={() => toggleLLMSelection(llm.id)}
                          className={cn("flex items-center gap-3 px-2 py-2 cursor-pointer rounded-md", isSelected ? "bg-[#0052CC]/10" : "hover:bg-muted")}
                        >
                          <Checkbox checked={isSelected} className="pointer-events-none" />
                          <div className="flex-1">
                            <p className="text-sm font-medium">{llm.name}</p>
                            <p className="text-xs text-muted-foreground">{llm.provider}</p>
                          </div>
                        </div>
                      )
                    })}
                  </DropdownMenuContent>
                </DropdownMenu>

                <Button
                  type="submit"
                  size="sm"
                  disabled={!input.trim() || isLoading || (!selectedModel && !sessionId)}
                  className={cn("h-8 w-8 rounded-full p-0", input.trim() && (selectedModel || sessionId) ? "bg-[#0052CC] hover:bg-[#003D99] text-white" : "bg-muted text-muted-foreground")}
                >
                  <ArrowUp className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </form>
          <p className="text-center text-xs text-muted-foreground mt-2">v.Alpha: Outputs may be incorrect, verify important information.</p>
        </div>
      </div>
    </TooltipProvider>
  )
}
