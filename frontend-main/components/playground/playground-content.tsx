"use client"

import type React from "react"
import { useState, useEffect, useRef, useCallback } from "react"
import { useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
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
import { mockModels } from "@/lib/mock-data"
import type { Model, LLMProvider, PlaygroundMessage, LLMOption, PlaygroundSession } from "@/lib/types"
import {
  Loader2,
  User,
  Database,
  GitCompare,
  X,
  ChevronDown,
  Plus,
  ArrowUp,
  Box,
  Clock,
  Zap,
  Check,
} from "lucide-react"
import { cn } from "@/lib/utils"
import {
  ResponseRenderer,
  type ResponseBlock,
  exampleTextResponse,
  exampleAnalyticsResponse,
} from "./response-renderer"

const llmOptions: LLMOption[] = [
  { id: "claude", name: "Claude 3.5 Sonnet", version: "3.5", provider: "Anthropic" },
  { id: "gpt-4", name: "GPT-4 Turbo", version: "4", provider: "OpenAI" },
  { id: "gemini", name: "Gemini Pro", version: "1.5", provider: "Google" },
  { id: "llama", name: "Llama 3.1", version: "70B", provider: "Meta" },
]

interface ExtendedPlaygroundMessage extends PlaygroundMessage {
  timeSpent?: number
  tokensBurned?: number
  groupId?: string
  richContent?: ResponseBlock[]
}

// Store for session chat histories
const sessionHistories: Record<string, {
  models: Model[]
  llms: LLMProvider[]
  messages: ExtendedPlaygroundMessage[]
  compareMode: boolean
}> = {}

export function PlaygroundContent() {
  const searchParams = useSearchParams()
  const modelIdFromUrl = searchParams.get("model")
  const newChatTrigger = searchParams.get("new")
  const sessionId = searchParams.get("session")
  const { addChatSession, chatSessions } = useSidebar()

  // State
  const [selectedModels, setSelectedModels] = useState<Model[]>([])
  const [compareMode, setCompareMode] = useState(false)
  const [selectedLLMs, setSelectedLLMs] = useState<LLMProvider[]>(["claude"])
  const [messages, setMessages] = useState<ExtendedPlaygroundMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const [llmDropdownOpen, setLlmDropdownOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Use refs to avoid dependency cycles in saveCurrentSession
  const selectedModelsRef = useRef(selectedModels)
  const selectedLLMsRef = useRef(selectedLLMs)
  const messagesRef = useRef(messages)
  const compareModeRef = useRef(compareMode)
  const currentSessionIdRef = useRef(currentSessionId)

  // Keep refs in sync
  useEffect(() => { selectedModelsRef.current = selectedModels }, [selectedModels])
  useEffect(() => { selectedLLMsRef.current = selectedLLMs }, [selectedLLMs])
  useEffect(() => { messagesRef.current = messages }, [messages])
  useEffect(() => { compareModeRef.current = compareMode }, [compareMode])
  useEffect(() => { currentSessionIdRef.current = currentSessionId }, [currentSessionId])

  // Save current session state before switching - uses refs to avoid dependency cycle
  const saveCurrentSession = useCallback(() => {
    if (currentSessionIdRef.current && messagesRef.current.length > 0) {
      sessionHistories[currentSessionIdRef.current] = {
        models: selectedModelsRef.current,
        llms: selectedLLMsRef.current,
        messages: messagesRef.current,
        compareMode: compareModeRef.current,
      }
    }
  }, [])

  // Load session when switching
  useEffect(() => {
    if (sessionId && sessionId !== currentSessionIdRef.current) {
      // Save previous session
      saveCurrentSession()

      // Check if we have saved history for this session
      const savedHistory = sessionHistories[sessionId]
      if (savedHistory) {
        setSelectedModels(savedHistory.models)
        setSelectedLLMs(savedHistory.llms)
        setMessages(savedHistory.messages)
        setCompareMode(savedHistory.compareMode)
      } else {
        // Check for pre-saved example sessions
        const session = chatSessions.find(s => s.id === sessionId)
        if (session) {
          // Load model
          const model = mockModels.find(m => session.modelIds.includes(m.id))
          if (model) {
            setSelectedModels([model])
          } else if (mockModels.length > 0) {
            setSelectedModels([mockModels[0]])
          }
          setSelectedLLMs(session.llmIds.length > 0 ? session.llmIds : ["claude"])
          setCompareMode(false)
          
          // Pre-populate with example content for Analysis and Analytics sessions
          if (session.id === "session-analysis") {
            const groupId = Date.now().toString()
            setMessages([
              {
                id: `${groupId}-user`,
                role: "user",
                content: "Show all text, table, code, and document examples",
                timestamp: new Date(),
                groupId,
              },
              {
                id: `${groupId}-response`,
                role: "assistant",
                content: "[Claude 3.5 Sonnet] Here's a comprehensive response with multiple data formats:",
                modelId: model?.id || mockModels[0]?.id,
                llmId: "claude",
                timestamp: new Date(),
                timeSpent: 1245,
                tokensBurned: 542,
                groupId,
                richContent: exampleTextResponse,
              },
            ])
          } else if (session.id === "session-analytics") {
            const groupId = Date.now().toString()
            setMessages([
              {
                id: `${groupId}-user`,
                role: "user",
                content: "Show analytics and charts",
                timestamp: new Date(),
                groupId,
              },
              {
                id: `${groupId}-response`,
                role: "assistant",
                content: "[Claude 3.5 Sonnet] Here's a data analytics overview with visualizations:",
                modelId: model?.id || mockModels[0]?.id,
                llmId: "claude",
                timestamp: new Date(),
                timeSpent: 1876,
                tokensBurned: 723,
                groupId,
                richContent: exampleAnalyticsResponse,
              },
            ])
          } else {
            setMessages([])
          }
        }
      }
      setCurrentSessionId(sessionId)
    }
  }, [sessionId, chatSessions, saveCurrentSession])

  // Reset everything when "new" param changes
  useEffect(() => {
    if (newChatTrigger) {
      saveCurrentSession()
      setSelectedModels([])
      setCompareMode(false)
      setSelectedLLMs(["claude"])
      setMessages([])
      setInput("")
      setCurrentSessionId(null)
    }
  }, [newChatTrigger, saveCurrentSession])

  // Initialize model from URL
  useEffect(() => {
    if (modelIdFromUrl && !newChatTrigger && !sessionId) {
      const model = mockModels.find((m) => m.id === modelIdFromUrl)
      if (model) {
        setSelectedModels([model])
        setCompareMode(false)
      }
    }
  }, [modelIdFromUrl, newChatTrigger, sessionId])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Check if model selection should be locked (has messages and not in compare mode)
  const isModelLocked = messages.length > 0 && !compareMode

  // Model selection handlers
  const toggleModelSelection = (model: Model) => {
    // Don't allow model changes if locked (has messages and not in compare mode)
    if (isModelLocked) {
      setModelDropdownOpen(false)
      return
    }
    
    if (compareMode) {
      const isSelected = selectedModels.some((m) => m.id === model.id)
      if (isSelected) {
        if (selectedModels.length > 1) {
          setSelectedModels(selectedModels.filter((m) => m.id !== model.id))
        }
      } else if (selectedModels.length < 4) {
        setSelectedModels([...selectedModels, model])
      }
    } else {
      setSelectedModels([model])
      setModelDropdownOpen(false)
    }
  }

  const toggleCompareMode = () => {
    if (!compareMode) {
      setSelectedLLMs(["claude"])
    } else {
      setSelectedModels(selectedModels.length > 0 ? [selectedModels[0]] : [])
    }
    setCompareMode(!compareMode)
    setMessages([])
  }

  // LLM selection handlers
  const toggleLLMSelection = (llmId: LLMProvider) => {
    if (compareMode) {
      setSelectedLLMs([llmId])
      setLlmDropdownOpen(false)
      return
    }

    const isSelected = selectedLLMs.includes(llmId)
    if (isSelected) {
      if (selectedLLMs.length > 1) {
        setSelectedLLMs(selectedLLMs.filter((id) => id !== llmId))
      }
    } else if (selectedLLMs.length < 2) {
      setSelectedLLMs([...selectedLLMs, llmId])
    }
  }

  // Detect query type and generate appropriate response
  const generateResponse = (model: Model, llm: LLMProvider, query: string): { content: string; richContent?: ResponseBlock[]; timeSpent: number; tokensBurned: number } => {
    const llmOption = llmOptions.find((l) => l.id === llm)
    const timeSpent = Math.floor(Math.random() * 2000) + 500
    const tokensBurned = Math.floor(Math.random() * 800) + 200
    const lowerQuery = query.toLowerCase()

    // Check for example queries that show rich content
    if (lowerQuery.includes("show all text") || lowerQuery.includes("table example") || lowerQuery.includes("code example") || lowerQuery.includes("document example")) {
      return {
        content: `[${llmOption?.name}] Here's a comprehensive response with multiple data formats:`,
        richContent: exampleTextResponse,
        timeSpent,
        tokensBurned,
      }
    }

    if (lowerQuery.includes("show analytics") || lowerQuery.includes("chart") || lowerQuery.includes("graph") || lowerQuery.includes("prediction") || lowerQuery.includes("metrics")) {
      return {
        content: `[${llmOption?.name}] Here's a data analytics overview with visualizations:`,
        richContent: exampleAnalyticsResponse,
        timeSpent,
        tokensBurned,
      }
    }

    // Generate table response for data queries
    if (lowerQuery.includes("top") || lowerQuery.includes("list") || lowerQuery.includes("show") || lowerQuery.includes("customers") || lowerQuery.includes("data")) {
      return {
        content: `[${llmOption?.name}] Based on my analysis of ${model.datasets.map((d) => d.datasetName).join(", ")}:`,
        richContent: [
          {
            type: "text",
            content: `I've queried your connected data sources and found the following relevant information. The analysis covers ${model.datasets.length} data source(s) with high correlation to your query.`,
          },
          {
            type: "table",
            title: "Query Results",
            content: {
              headers: ["ID", "Name", "Value", "Status", "Updated"],
              rows: [
                ["001", "Primary Record", "$45,230", "Active", "Today"],
                ["002", "Secondary Record", "$38,100", "Active", "Yesterday"],
                ["003", "Tertiary Record", "$29,450", "Pending", "2 days ago"],
                ["004", "Fourth Record", "$21,800", "Active", "3 days ago"],
                ["005", "Fifth Record", "$18,200", "Inactive", "1 week ago"],
              ],
            },
          },
        ],
        timeSpent,
        tokensBurned,
      }
    }

    // Default text response
    return {
      content: `[${llmOption?.name}] Based on my analysis of ${model.datasets.map((d) => d.datasetName).join(", ")}, I can provide the following insights:\n\nThe analysis suggests key patterns in your data that align with the query parameters. The model has processed the relevant columns and identified correlations across the connected data sources.`,
      timeSpent,
      tokensBurned,
    }
  }

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading || selectedModels.length === 0) return

    const userMessageContent = input.trim()

    // Create session on first message
    if (!currentSessionId && messages.length === 0) {
      const modelName = selectedModels[0]?.name || "Chat"
      const dataSourceName = selectedModels[0]?.datasets[0]?.datasetName || ""
      const sessionName = dataSourceName
        ? `${modelName} - ${dataSourceName}`.slice(0, 40)
        : modelName.slice(0, 40)

      const newSession: PlaygroundSession = {
        id: `session-${Date.now()}`,
        name: sessionName,
        modelIds: selectedModels.map((m) => m.id),
        llmIds: selectedLLMs,
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      setCurrentSessionId(newSession.id)
      addChatSession(newSession)
    }

    const groupId = Date.now().toString()

    const userMessage: ExtendedPlaygroundMessage = {
      id: groupId,
      role: "user",
      content: userMessageContent,
      timestamp: new Date(),
      groupId,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }

    setTimeout(() => {
      const newResponses: ExtendedPlaygroundMessage[] = []

      if (compareMode) {
        selectedModels.forEach((model) => {
          const response = generateResponse(model, selectedLLMs[0], userMessageContent)
          newResponses.push({
            id: `${groupId}-${model.id}`,
            role: "assistant",
            content: response.content,
            modelId: model.id,
            llmId: selectedLLMs[0],
            timestamp: new Date(),
            timeSpent: response.timeSpent,
            tokensBurned: response.tokensBurned,
            richContent: response.richContent,
            groupId,
          })
        })
      } else if (selectedLLMs.length > 1) {
        selectedLLMs.forEach((llm) => {
          const response = generateResponse(selectedModels[0], llm, userMessageContent)
          newResponses.push({
            id: `${groupId}-${llm}`,
            role: "assistant",
            content: response.content,
            modelId: selectedModels[0].id,
            llmId: llm,
            timestamp: new Date(),
            timeSpent: response.timeSpent,
            tokensBurned: response.tokensBurned,
            richContent: response.richContent,
            groupId,
          })
        })
      } else {
        const response = generateResponse(selectedModels[0], selectedLLMs[0], userMessageContent)
        newResponses.push({
          id: `${groupId}-response`,
          role: "assistant",
          content: response.content,
          modelId: selectedModels[0].id,
          llmId: selectedLLMs[0],
          timestamp: new Date(),
          timeSpent: response.timeSpent,
          tokensBurned: response.tokensBurned,
          richContent: response.richContent,
          groupId,
        })
      }

      setMessages((prev) => [...prev, ...newResponses])
      setIsLoading(false)
    }, 1500)
  }

  // Connected data info
  const datasetsMap = new Map()
  selectedModels.forEach((model) => {
    model.datasets.forEach((ds) => {
      if (!datasetsMap.has(ds.datasetId)) {
        datasetsMap.set(ds.datasetId, ds)
      }
    })
  })
  const connectedData = Array.from(datasetsMap.values())

  // Group messages by groupId
  const groupedMessages: { user: ExtendedPlaygroundMessage; responses: ExtendedPlaygroundMessage[] }[] = []
  const seenGroups = new Set<string>()

  messages.forEach((msg) => {
    if (msg.role === "user" && msg.groupId && !seenGroups.has(msg.groupId)) {
      seenGroups.add(msg.groupId)
      const responses = messages.filter((m) => m.role === "assistant" && m.groupId === msg.groupId)
      groupedMessages.push({ user: msg, responses })
    }
  })

  // No model selected - show model selection screen
  if (selectedModels.length === 0) {
    return (
      <div className="flex h-[calc(100vh-6rem)] flex-col items-center justify-center px-4">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#0052CC]/20 to-[#003D99]/20 dark:from-[#2684FF]/20 dark:to-[#0052CC]/20">
          <Box className="h-8 w-8 text-[#0052CC] dark:text-[#2684FF]" />
        </div>
        <h3 className="mt-4 text-lg font-medium text-foreground">Select a Model to Start</h3>
        <p className="mt-2 max-w-md text-center text-sm text-muted-foreground">
          Choose one of your built models to start chatting with your data
        </p>

        <div className="mt-8 w-full max-w-md space-y-3">
          {mockModels.length === 0 ? (
            <div className="text-center">
              <p className="text-sm text-muted-foreground">No models built yet.</p>
              <Button
                className="mt-4 bg-[#0052CC] hover:bg-[#003D99] text-white"
                onClick={() => (window.location.href = "/build")}
              >
                Build Your First Model
              </Button>
            </div>
          ) : (
            mockModels.map((model) => (
              <button
                key={model.id}
                onClick={() => setSelectedModels([model])}
                className="w-full flex items-center gap-4 rounded-xl border border-border bg-card p-4 text-left transition-all hover:border-[#0052CC]/30 hover:bg-[#0052CC]/5 dark:hover:border-[#2684FF]/30 dark:hover:bg-[#2684FF]/5"
              >
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#2684FF]/20">
                  <Box className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-medium text-foreground truncate">{model.name}</h4>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {model.datasets.length} data source{model.datasets.length !== 1 ? "s" : ""} connected
                  </p>
                </div>
                <div className="text-xs text-muted-foreground">{(model.accuracy * 100).toFixed(0)}% accuracy</div>
              </button>
            ))
          )}
        </div>
      </div>
    )
  }

  return (
    <TooltipProvider>
      <div className="flex h-[calc(100vh-6rem)] flex-col">
        {/* Header Controls */}
        <div className="shrink-0 border-b border-border px-4 py-3">
          <div className="flex items-center justify-between gap-4">
            {/* Left: Model selection */}
            <div className="flex items-center gap-3">
              <DropdownMenu open={modelDropdownOpen} onOpenChange={isModelLocked ? undefined : setModelDropdownOpen}>
                <DropdownMenuTrigger asChild>
                  <Button 
                    variant="outline" 
                    className={cn(
                      "gap-2 border-border bg-card",
                      isModelLocked && "cursor-default opacity-70"
                    )}
                    disabled={isModelLocked}
                  >
                    <Box className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />
                    <span className="truncate max-w-[150px]">
                      {compareMode
                        ? `${selectedModels.length} Model${selectedModels.length !== 1 ? "s" : ""}`
                        : selectedModels[0]?.name || "Select Model"}
                    </span>
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-64">
                  <DropdownMenuLabel className="text-xs text-muted-foreground">
                    {compareMode ? "Select up to 4 models" : "Select a model"}
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {mockModels.map((model) => {
                    const isSelected = selectedModels.some((m) => m.id === model.id)
                    return (
                      <div
                        key={model.id}
                        onClick={() => toggleModelSelection(model)}
                        className={cn(
                          "flex items-center gap-3 px-2 py-2 cursor-pointer rounded-md transition-colors",
                          isSelected ? "bg-[#0052CC]/10 dark:bg-[#2684FF]/10" : "hover:bg-muted"
                        )}
                      >
                        {compareMode && <Checkbox checked={isSelected} className="pointer-events-none" />}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{model.name}</p>
                          <p className="text-xs text-muted-foreground">{model.datasets.length} sources</p>
                        </div>
                        {!compareMode && isSelected && <Check className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />}
                      </div>
                    )
                  })}
                </DropdownMenuContent>
              </DropdownMenu>

              {/* Compare mode selected models chips */}
              {compareMode && selectedModels.length > 1 && (
                <div className="flex items-center gap-1.5 flex-wrap">
                  {selectedModels.slice(1).map((model) => (
                    <div
                      key={model.id}
                      className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs bg-muted text-muted-foreground"
                    >
                      {model.name}
                      <button
                        onClick={() => setSelectedModels(selectedModels.filter((m) => m.id !== model.id))}
                        className="hover:opacity-70"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Connected data tooltip */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <button className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
                    <Database className="h-3.5 w-3.5" />
                    <span>
                      {connectedData.length} source{connectedData.length !== 1 ? "s" : ""}
                    </span>
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-xs p-3 bg-popover border border-border">
                  <p className="text-xs font-medium mb-2 text-foreground">Connected Data Sources</p>
                  <div className="space-y-1.5">
                    {connectedData.map((ds) => (
                      <div key={ds.datasetId} className="flex items-center gap-2 text-xs">
                        <SourceBadge source={ds.source} size="sm" />
                        <span className="text-foreground">{ds.datasetName}</span>
                      </div>
                    ))}
                  </div>
                </TooltipContent>
              </Tooltip>
            </div>

            {/* Right: Compare mode toggle */}
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

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center px-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted">
                <Box className="h-6 w-6 text-muted-foreground" />
              </div>
              <p className="mt-4 text-sm text-muted-foreground">Start chatting with your data</p>
            </div>
          ) : (
            <div className="mx-auto max-w-4xl p-4 space-y-6">
              {groupedMessages.map((group) => (
                <div key={group.user.groupId} className="space-y-4">
                  {/* User message */}
                  <div className="flex justify-end">
                    <div className="flex items-start gap-3 max-w-[80%]">
                      <div className="rounded-2xl rounded-tr-md bg-[#0052CC] px-4 py-2.5 text-white">
                        <p className="text-sm">{group.user.content}</p>
                      </div>
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                        <User className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  </div>

                  {/* Responses - side by side for multiple */}
                  {group.responses.length > 1 ? (
                    <div className={cn("grid gap-4", group.responses.length === 2 && "grid-cols-2", group.responses.length === 3 && "grid-cols-3", group.responses.length === 4 && "grid-cols-2 lg:grid-cols-4")}>
                      {group.responses.map((response) => {
                        const llm = llmOptions.find((l) => l.id === response.llmId)
                        const model = mockModels.find((m) => m.id === response.modelId)
                        return (
                          <div key={response.id} className="space-y-2">
                            <div className="flex items-center gap-2">
                              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-muted">
                                <Box className="h-3 w-3 text-muted-foreground" />
                              </div>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                                {compareMode ? model?.name : llm?.name}
                              </span>
                            </div>
                            <div className="rounded-2xl rounded-tl-md border border-border bg-card p-4">
                              <p className="text-sm text-foreground whitespace-pre-wrap">{response.content}</p>
                              {response.richContent && (
                                <div className="mt-4">
                                  <ResponseRenderer blocks={response.richContent} />
                                </div>
                              )}
                              <div className="flex items-center gap-3 mt-3 pt-3 border-t border-border">
                                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                  <Clock className="h-3 w-3" />
                                  {(response.timeSpent! / 1000).toFixed(2)}s
                                </span>
                                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                  <Zap className="h-3 w-3" />
                                  {response.tokensBurned} tokens
                                </span>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    group.responses.map((response) => {
                      const llm = llmOptions.find((l) => l.id === response.llmId)
                      return (
                        <div key={response.id} className="flex items-start gap-3">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                            <Box className="h-4 w-4 text-muted-foreground" />
                          </div>
                          <div className="flex-1 space-y-2">
                            <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                              {llm?.name}
                            </span>
                            <div className="rounded-2xl rounded-tl-md border border-border bg-card p-4">
                              <p className="text-sm text-foreground whitespace-pre-wrap">{response.content}</p>
                              {response.richContent && (
                                <div className="mt-4">
                                  <ResponseRenderer blocks={response.richContent} />
                                </div>
                              )}
                              <div className="flex items-center gap-3 mt-3 pt-3 border-t border-border">
                                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                  <Clock className="h-3 w-3" />
                                  {(response.timeSpent! / 1000).toFixed(2)}s
                                </span>
                                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                  <Zap className="h-3 w-3" />
                                  {response.tokensBurned} tokens
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )
                    })
                  )}
                </div>
              ))}

              {isLoading && (
                <div className="flex items-start gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
                    <Box className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="rounded-2xl rounded-tl-md border border-border bg-card p-4">
                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="shrink-0 border-t border-border p-4">
          <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
            <div className="rounded-2xl border border-border bg-card p-3">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about your data..."
                className="min-h-[44px] max-h-[200px] resize-none border-0 bg-transparent p-0 text-sm focus-visible:ring-0 placeholder:text-muted-foreground"
                disabled={isLoading}
                rows={1}
              />
              <div className="mt-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button type="button" variant="ghost" size="sm" className="h-8 w-8 p-0 text-muted-foreground">
                        <Plus className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      <p>Custom script</p>
                    </TooltipContent>
                  </Tooltip>

                  {/* LLM Dropdown - multi-select in non-compare mode */}
                  <DropdownMenu open={llmDropdownOpen} onOpenChange={setLlmDropdownOpen}>
                    <DropdownMenuTrigger asChild>
                      <Button type="button" variant="ghost" size="sm" className="h-8 gap-1.5 px-2 text-muted-foreground">
                        <span className="text-xs">
                          {selectedLLMs.length === 1
                            ? llmOptions.find((l) => l.id === selectedLLMs[0])?.name
                            : `${selectedLLMs.length} LLMs`}
                        </span>
                        <ChevronDown className="h-3 w-3" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start" className="w-56">
                      <DropdownMenuLabel className="text-xs text-muted-foreground">
                        {compareMode ? "Select LLM (single only in compare mode)" : "Select up to 2 LLMs"}
                      </DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      {llmOptions.map((llm) => {
                        const isSelected = selectedLLMs.includes(llm.id)
                        return (
                          <div
                            key={llm.id}
                            onClick={() => toggleLLMSelection(llm.id)}
                            className={cn(
                              "flex items-center gap-3 px-2 py-2 cursor-pointer rounded-md transition-colors",
                              isSelected ? "bg-[#0052CC]/10 dark:bg-[#2684FF]/10" : "hover:bg-muted"
                            )}
                          >
                            {!compareMode && <Checkbox checked={isSelected} className="pointer-events-none" />}
                            <div className="flex-1">
                              <p className="text-sm font-medium">{llm.name}</p>
                              <p className="text-xs text-muted-foreground">{llm.provider}</p>
                            </div>
                            {compareMode && isSelected && <Check className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />}
                          </div>
                        )
                      })}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                <Button
                  type="submit"
                  size="sm"
                  disabled={!input.trim() || isLoading}
                  className={cn(
                    "h-8 w-8 rounded-full p-0 transition-colors",
                    input.trim()
                      ? "bg-[#0052CC] hover:bg-[#003D99] text-white"
                      : "bg-muted text-muted-foreground"
                  )}
                >
                  <ArrowUp className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </TooltipProvider>
  )
}
