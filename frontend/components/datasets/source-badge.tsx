import type React from "react"
import { cn } from "@/lib/utils"
import type { DataSource } from "@/lib/types"
import { Database, Cloud, Upload, HardDrive, Server, Sparkles } from "lucide-react"

interface SourceBadgeProps {
  source: DataSource
  size?: "sm" | "md"
  showLabel?: boolean
}

const sourceConfig: Record<DataSource, { label: string; color: string; icon: React.ReactNode }> = {
  "uploaded-files": { label: "Uploaded Files", color: "bg-blue-500/10 text-blue-500", icon: <Upload className="h-3 w-3" /> },
  databricks: {
    label: "Databricks",
    color: "bg-orange-500/20 text-orange-400 border-orange-500/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  supabase: {
    label: "Supabase",
    color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  api: {
    label: "API",
    color: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    icon: <Cloud className="h-3.5 w-3.5" />,
  },
  "google-drive": {
    label: "Google Drive",
    color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    icon: <HardDrive className="h-3.5 w-3.5" />,
  },
  postgresql: {
    label: "PostgreSQL",
    color: "bg-blue-600/20 text-blue-500 border-blue-600/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  mongodb: {
    label: "MongoDB",
    color: "bg-green-600/20 text-green-500 border-green-600/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  snowflake: {
    label: "Snowflake",
    color: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  pinecone: {
    label: "Pinecone",
    color: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  gcs: {
    label: "Google Cloud Storage",
    color: "bg-red-500/20 text-red-400 border-red-500/30",
    icon: <Cloud className="h-3.5 w-3.5" />,
  },
  "aws-s3": {
    label: "AWS S3",
    color: "bg-orange-600/20 text-orange-500 border-orange-600/30",
    icon: <Cloud className="h-3.5 w-3.5" />,
  },
  mysql: {
    label: "MySQL",
    color: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  weaviate: {
    label: "Weaviate",
    color: "bg-teal-500/20 text-teal-400 border-teal-500/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  chroma: {
    label: "Chroma",
    color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  lancedb: {
    label: "LanceDB",
    color: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
    icon: <Database className="h-3.5 w-3.5" />,
  },
  graphql: {
    label: "GraphQL",
    color: "bg-pink-500/20 text-pink-400 border-pink-500/30",
    icon: <Cloud className="h-3.5 w-3.5" />,
  },
  rest: {
    label: "REST API",
    color: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
    icon: <Cloud className="h-3.5 w-3.5" />,
  },
  upload: {
    label: "Upload",
    color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    icon: <Upload className="h-3.5 w-3.5" />,
  },
  generated: {
    label: "Generated",
    color: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    icon: <Sparkles className="h-3.5 w-3.5" />,
  },
}

export function SourceBadge({ source, size = "md", showLabel = true }: SourceBadgeProps) {
  const config = sourceConfig[source] || sourceConfig.upload

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border",
        config.color,
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs"
      )}
    >
      {config.icon}
      {showLabel && <span className="font-medium">{config.label}</span>}
    </div>
  )
}
