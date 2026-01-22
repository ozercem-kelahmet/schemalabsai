import type React from "react"
import { cn } from "@/lib/utils"
import type { DataSource } from "@/lib/types"

interface SourceBadgeProps {
  source: DataSource
  size?: "sm" | "md"
  showLabel?: boolean
}

const sourceConfig: Record<DataSource, { label: string; color: string; icon: React.ReactNode }> = {
  databricks: {
    label: "Databricks",
    color: "bg-orange-500/20 text-orange-400 border-orange-500/30",
    icon: (
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
  },
  supabase: {
    label: "Supabase",
    color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    icon: (
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor">
        <path d="M21.362 9.354H12V.396a.396.396 0 00-.716-.233L2.203 12.424l-.401.562a1.04 1.04 0 00.836 1.659H12v8.959a.396.396 0 00.716.233l9.081-12.261.401-.562a1.04 1.04 0 00-.836-1.66z" />
      </svg>
    ),
  },
  api: {
    label: "API",
    color: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    icon: (
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 6h16M4 12h16M4 18h16" />
      </svg>
    ),
  },
  "google-drive": {
    label: "Google Drive",
    color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    icon: (
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor">
        <path d="M12 2L4 14h4l4-7 4 7h4L12 2zM4 14l4 8h8l4-8H4z" />
      </svg>
    ),
  },
}

const fallbackConfig = {
  label: "Unknown",
  color: "bg-gray-500/20 text-gray-400 border-gray-500/30",
  icon: (
    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  ),
}

export function SourceBadge({ source, size = "md", showLabel = true }: SourceBadgeProps) {
  const config = sourceConfig[source] || fallbackConfig

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border font-medium",
        config.color,
        size === "sm" ? "px-1.5 py-0.5 text-[10px]" : "px-2 py-1 text-xs",
      )}
    >
      {config.icon}
      {showLabel && <span>{config.label}</span>}
    </div>
  )
}
