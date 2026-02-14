"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, Loader2, Sparkles, User, Database } from "lucide-react"
import type { Model } from "@/lib/types"
import { SourceBadge } from "@/components/datasets/source-badge"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  prediction?: {
    result: string
    confidence: number
    details?: Record<string, unknown>
  }
}

interface ChatInterfaceProps {
  model: Model
  onQuery: (input: string) => void
}

export function ChatInterface({ model, onQuery }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Reset messages when model changes
  useEffect(() => {
    setMessages([])
  }, [model.id])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const generateResponse = () => {
    const responses = [
      `Based on the ${model.datasets.length} connected data source(s), I can provide insights about this query.`,
      `Analyzing data from ${model.datasets.map((d) => d.datasetName).join(", ")}...`,
      `Using the ${model.modelType.replace("-", " ")} capabilities to process your request.`,
    ]

    const isPositive = Math.random() > 0.5
    const confidence = 0.75 + Math.random() * 0.2

    return {
      content: responses[Math.floor(Math.random() * responses.length)],
      prediction: {
        result: isPositive ? "Positive" : "Negative",
        confidence,
        details: {
          probability: confidence,
          topFactors: model.datasets.flatMap((d) => d.datasetName.split(" ").slice(0, 2)).slice(0, 3),
        },
      },
    }
  }

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)
    onQuery(input)

    setTimeout(() => {
      const response = generateResponse()
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.content,
        prediction: response.prediction,
      }
      setMessages((prev) => [...prev, assistantMessage])
      setIsLoading(false)
    }, 1500)
  }

  const exampleQueries = [
    `What insights can you provide from the ${model.datasets[0]?.datasetName || "connected data"}?`,
    "Analyze the key patterns in the data",
    "What are the main factors affecting outcomes?",
  ]

  const handleExampleClick = (example: string) => {
    setInput(example)
  }

  return (
    <div className="flex h-[calc(100vh-200px)] flex-col rounded-xl border border-white/10 bg-[#111113]">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#2684FF]/20 to-[#0052CC]/20">
              <Sparkles className="h-8 w-8 text-[#2684FF]" />
            </div>
            <h3 className="mt-4 text-lg font-medium text-white">Chat with {model.name}</h3>
            <p className="mt-2 max-w-md text-center text-sm text-gray-400">{model.description}</p>

            <div className="mt-4 flex flex-wrap justify-center gap-2">
              <span className="flex items-center gap-1.5 text-xs text-gray-500">
                <Database className="h-3 w-3" />
                Connected to:
              </span>
              {model.datasets.map((ds) => (
                <div key={ds.datasetId} className="flex items-center gap-1.5 rounded bg-white/5 px-2 py-1">
                  <SourceBadge source={ds.source} size="sm" />
                  <span className="text-xs text-gray-400">{ds.datasetName}</span>
                </div>
              ))}
            </div>

            {/* Example Queries */}
            <div className="mt-6 space-y-2">
              <p className="text-xs text-gray-500">Try an example:</p>
              {exampleQueries.map((example, i) => (
                <button
                  key={i}
                  onClick={() => handleExampleClick(example)}
                  className="block w-full rounded-lg border border-white/10 bg-white/5 px-4 py-2.5 text-left text-sm text-gray-300 transition-colors hover:border-[#0052CC]/30 hover:bg-[#0052CC]/10 hover:text-white"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div key={message.id} className={`flex gap-3 ${message.role === "user" ? "justify-end" : ""}`}>
                {message.role === "assistant" && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[#2684FF] to-[#0052CC]">
                    <Sparkles className="h-4 w-4 text-white" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] rounded-xl px-4 py-3 ${
                    message.role === "user" ? "bg-[#0052CC]/20 text-white" : "bg-white/5 text-gray-300"
                  }`}
                >
                  <p className="text-sm">{message.content}</p>

                  {message.prediction && (
                    <div className="mt-3 rounded-lg bg-black/30 p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500">Analysis Result</span>
                        <span
                          className={`rounded px-2 py-0.5 text-xs font-medium ${
                            message.prediction.result === "Positive"
                              ? "bg-emerald-500/20 text-emerald-400"
                              : "bg-red-500/20 text-red-400"
                          }`}
                        >
                          {message.prediction.result}
                        </span>
                      </div>
                      <div className="mt-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-gray-400">Confidence</span>
                          <span className="font-mono text-[#2684FF]">
                            {(message.prediction.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full bg-[#0052CC]"
                            style={{ width: `${message.prediction.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                      {message.prediction.details && (
                        <div className="mt-3 border-t border-white/5 pt-3">
                          <p className="text-xs text-gray-500">Key Factors</p>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {(message.prediction.details.topFactors as string[]).map((factor) => (
                              <span
                                key={factor}
                                className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-xs text-gray-400"
                              >
                                {factor}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                {message.role === "user" && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-gray-600 to-gray-700">
                    <User className="h-4 w-4 text-white" />
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-[#2684FF] to-[#0052CC]">
                  <Sparkles className="h-4 w-4 text-white" />
                </div>
                <div className="rounded-xl bg-white/5 px-4 py-3">
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing...
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-white/10 p-4">
        <div className="flex gap-3">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your data..."
            className="min-h-[44px] max-h-32 resize-none border-white/10 bg-white/5 text-white placeholder:text-gray-500"
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                handleSubmit()
              }
            }}
          />
          <Button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className="h-11 w-11 shrink-0 bg-[#0052CC] p-0 text-white hover:bg-[#003D99] disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </div>
  )
}
