"use client"

import type React from "react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import { Mic, Paperclip, Plus, Search, Send, Sparkles, Database, ChevronDown } from "lucide-react"
import { useRef, useState, useEffect } from "react"

const MODELS = [
  { value: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", description: "Fast and intelligent" },
  { value: "claude-opus-4", name: "Claude Opus 4", description: "Most capable" },
  { value: "claude-haiku-4-5", name: "Claude Haiku 4.5", description: "Lightning fast" },
  { value: "gpt-4o", name: "GPT-4o", description: "Fast and capable" },
]

interface AiChatProps {
  selectedModel?: string
  onSend?: (message: string, model: string) => void
  dataSourceName?: string
  dataSources?: string[]
}

export default function AiChat({ onSend, dataSourceName, dataSources = [], selectedModel: propModel }: AiChatProps) {
  const [message, setMessage] = useState("")
  const [isExpanded, setIsExpanded] = useState(false)
  const [selectedModel, setSelectedModel] = useState(() => {
    const found = MODELS.find(m => m.value === propModel)
    return found || MODELS[0]
  })

  useEffect(() => {
    if (propModel) {
      const found = MODELS.find(m => m.value === propModel)
      if (found) setSelectedModel(found)
    }
  }, [propModel])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (message.trim() && onSend) {
      onSend(message, selectedModel.value)
      setMessage("")
      setIsExpanded(false)

      if (textareaRef.current) {
        textareaRef.current.style.height = "auto"
      }
    }
  }

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value)

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }

    setIsExpanded(e.target.value.length > 100 || e.target.value.includes("\n"))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as React.FormEvent)
    }
  }

  const displayDataSources = dataSources.length > 0 ? dataSources : dataSourceName ? [dataSourceName] : []

  return (
    <div className="w-full max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="group/composer w-full">
        <input ref={fileInputRef} type="file" multiple className="sr-only" onChange={() => {}} />

        <div
          className={cn(
            "w-full bg-card cursor-text overflow-clip bg-clip-padding p-2.5 shadow-lg border border-border transition-all duration-200",
            {
              "rounded-3xl grid grid-cols-1 grid-rows-[auto_1fr_auto]": isExpanded,
              "rounded-[28px] grid grid-cols-[auto_1fr_auto] grid-rows-[auto_1fr_auto]": !isExpanded,
            },
          )}
          style={{
            gridTemplateAreas: isExpanded
              ? "'header' 'primary' 'footer'"
              : "'header header header' 'leading primary trailing' '. footer .'",
          }}
        >
          <div
            className={cn("flex min-h-14 items-center overflow-x-hidden px-1.5", {
              "px-2 py-1 mb-0": isExpanded,
              "-my-2.5": !isExpanded,
            })}
            style={{ gridArea: "primary" }}
          >
            <div className="flex-1 overflow-auto max-h-52">
              <Textarea
                ref={textareaRef}
                value={message}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything about your data..."
                className="min-h-0 resize-none rounded-none border-0 p-0 text-base placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 scrollbar-thin bg-transparent"
                rows={1}
              />
            </div>
          </div>

          <div className={cn("flex", { hidden: isExpanded })} style={{ gridArea: "leading" }}>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9 rounded-full hover:bg-accent outline-none ring-0"
                >
                  <Plus className="size-5 text-muted-foreground" />
                </Button>
              </DropdownMenuTrigger>

              <DropdownMenuContent align="start" className="max-w-xs rounded-2xl p-1.5">
                <DropdownMenuGroup className="space-y-1">
                  <DropdownMenuItem className="rounded-[calc(1rem-6px)]" onClick={() => fileInputRef.current?.click()}>
                    <Paperclip size={20} className="opacity-60" />
                    Add files
                  </DropdownMenuItem>
                  <DropdownMenuItem className="rounded-[calc(1rem-6px)]">
                    <Sparkles size={20} className="opacity-60" />
                    Agent mode
                  </DropdownMenuItem>
                  <DropdownMenuItem className="rounded-[calc(1rem-6px)]">
                    <Search size={20} className="opacity-60" />
                    Deep Research
                  </DropdownMenuItem>
                </DropdownMenuGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <div className="flex items-center gap-2" style={{ gridArea: isExpanded ? "footer" : "trailing" }}>
            <div className="ms-auto flex items-center gap-1.5">
              <Button type="button" variant="ghost" size="icon" className="h-9 w-9 rounded-full hover:bg-accent">
                <Mic className="size-5 text-muted-foreground" />
              </Button>

              {message.trim() && (
                <Button type="submit" size="icon" className="h-9 w-9 rounded-full">
                  <Send className="size-5" />
                </Button>
              )}
            </div>
          </div>

          <div
            className={cn("flex items-center gap-3 px-2 pt-2", { hidden: !isExpanded })}
            style={{ gridArea: "footer" }}
          >
            {/* Model selector */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground gap-1"
                >
                  {selectedModel.name}
                  <ChevronDown className="size-3" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="rounded-xl">
                {MODELS.map((model) => (
                  <DropdownMenuItem
                    key={model.value}
                    onClick={() => setSelectedModel(model)}
                    className="flex flex-col items-start"
                  >
                    <span className="font-medium">{model.name}</span>
                    <span className="text-xs text-muted-foreground">{model.description}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Data sources indicator */}
            {displayDataSources.length > 0 && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Database className="size-3" />
                <span className="truncate max-w-[150px]">
                  {displayDataSources.length === 1 ? displayDataSources[0] : `${displayDataSources.length} sources`}
                </span>
              </div>
            )}

            <div className="ml-auto flex items-center gap-1.5">
              <Button type="button" variant="ghost" size="icon" className="h-9 w-9 rounded-full hover:bg-accent">
                <Mic className="size-5 text-muted-foreground" />
              </Button>

              {message.trim() && (
                <Button type="submit" size="icon" className="h-9 w-9 rounded-full">
                  <Send className="size-5" />
                </Button>
              )}
            </div>
          </div>
        </div>

        {!isExpanded && displayDataSources.length > 0 && (
          <div className="flex items-center justify-center gap-2 mt-3">
            <Database className="size-3.5 text-muted-foreground" />
            <div className="flex items-center gap-1.5 flex-wrap justify-center">
              {displayDataSources.slice(0, 3).map((source, i) => (
                <span
                  key={i}
                  className="inline-flex items-center rounded-full bg-secondary/50 px-2.5 py-0.5 text-xs text-muted-foreground"
                >
                  {source}
                </span>
              ))}
              {displayDataSources.length > 3 && (
                <span className="text-xs text-muted-foreground">+{displayDataSources.length - 3} more</span>
              )}
            </div>
          </div>
        )}

        {!isExpanded && (
          <div className="flex justify-center mt-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-3 text-xs text-muted-foreground hover:text-foreground gap-1 rounded-full"
                >
                  {selectedModel.name}
                  <ChevronDown className="size-3" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="center" className="rounded-xl">
                {MODELS.map((model) => (
                  <DropdownMenuItem
                    key={model.value}
                    onClick={() => setSelectedModel(model)}
                    className="flex flex-col items-start"
                  >
                    <span className="font-medium">{model.name}</span>
                    <span className="text-xs text-muted-foreground">{model.description}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </form>
    </div>
  )
}
