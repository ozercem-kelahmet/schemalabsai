"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { SourceBadge } from "./source-badge"
import type { Dataset } from "@/lib/types"
import { Table, Columns, Pencil, Trash2, Check, X, Download } from "lucide-react"

interface DatasetCardProps {
  dataset: Dataset
  onViewSchema: (dataset: Dataset) => void
  onEdit?: (dataset: Dataset, newName: string) => void
  onDelete?: (dataset: Dataset) => void
}

const verticalLabels: Record<string, string> = {
  finance: "Finance",
  healthcare: "Healthcare",
  "e-commerce": "E-commerce",
  marketing: "Marketing",
  hr: "HR",
  operations: "Operations",
}

const uploadSources = ["upload", "google-drive", "generated"]

export function DatasetCard({ dataset, onViewSchema, onEdit, onDelete }: DatasetCardProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(dataset.name)

  const handleEditClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsEditing(true)
    setEditName(dataset.name)
  }

  const handleSave = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (editName.trim() && onEdit) {
      onEdit(dataset, editName.trim())
    }
    setIsEditing(false)
  }

  const handleCancel = (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsEditing(false)
    setEditName(dataset.name)
  }

  return (
    <Card
      className="group relative border-border bg-card transition-all hover:border-[#0052CC]/30 hover:bg-accent/50 cursor-pointer overflow-hidden"
      onClick={() => !isEditing && onViewSchema(dataset)}
    >
      <CardContent className="p-5">
        {/* Action buttons - show on hover */}
        {!isEditing && (
          <div className="absolute top-2 left-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
            {onEdit && (
              <button
                onClick={handleEditClick}
                className="p-1.5 rounded-md bg-background/80 backdrop-blur-sm border border-border hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                title="Edit"
              >
                <Pencil className="h-3.5 w-3.5" />
              </button>
            )}
            {/* Download button */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                const API_HOST = window.location.origin.includes(":3000") ? window.location.origin.replace(":3000", ":8080") : window.location.origin
                const link = document.createElement("a")
                link.href = API_HOST + "/api/download/" + dataset.id
                link.download = dataset.name + ".csv"
                document.body.appendChild(link)
                link.click()
                document.body.removeChild(link)
              }}
              className="p-1.5 rounded-md bg-background/80 backdrop-blur-sm border border-border hover:bg-[#0052CC]/10 text-muted-foreground hover:text-[#0052CC] transition-colors"
              title="Download CSV"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
            {onDelete && (
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(dataset); }}
                className="p-1.5 rounded-md bg-background/80 backdrop-blur-sm border border-border hover:bg-red-500/10 text-muted-foreground hover:text-red-500 transition-colors"
                title="Delete"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        )}

        {/* Header */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            {isEditing ? (
              <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="w-full min-w-0 px-2 py-1 text-sm bg-muted border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-[#0052CC]"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSave(e as any)
                    if (e.key === 'Escape') handleCancel(e as any)
                  }}
                />
                <button
                  onClick={handleSave}
                  className="p-1 rounded-md bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20 flex-shrink-0"
                  title="Save"
                >
                  <Check className="h-4 w-4" />
                </button>
                <button
                  onClick={handleCancel}
                  className="p-1 rounded-md bg-red-500/10 text-red-500 hover:bg-red-500/20 flex-shrink-0"
                  title="Cancel"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ) : (
              <>
                <h3 className="font-medium text-foreground group-hover:text-[#2684FF] transition-colors truncate">{dataset.name}</h3>
                <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{dataset.description}</p>
              </>
            )}
          </div>
          {!isEditing && <SourceBadge source={dataset.source} />}
        </div>

        {/* Tags */}
        <div className="mt-4 flex flex-wrap items-center gap-2">


          {verticalLabels[dataset.vertical] && (
            <span className="rounded-md bg-muted px-2 py-1 text-xs text-muted-foreground">
              {verticalLabels[dataset.vertical]}
            </span>
          )}
          <span className="rounded-md bg-muted px-2 py-1 text-xs capitalize text-muted-foreground">
            {dataset.complexity}
          </span>
          {dataset.syncStatus && (
            <span
              className={`rounded-md px-2 py-1 text-xs ${
                uploadSources.includes(dataset.source)
                  ? "bg-muted text-muted-foreground"
                  : dataset.syncStatus === "synced"
                    ? "bg-emerald-500/10 text-emerald-500"
                    : dataset.syncStatus === "pending"
                    ? "bg-yellow-500/10 text-yellow-500"
                    : "bg-orange-500/10 text-orange-500"
              }`}
            >
              {uploadSources.includes(dataset.source) ? "Static" : dataset.syncStatus === "synced" ? "Synced" : dataset.syncStatus === "pending" ? "Pending" : "Outdated"}
            </span>
          )}
        </div>

        {/* Stats */}
        <div className="mt-4 flex items-center gap-2 border-t border-border pt-4 text-xs text-muted-foreground">
          {(dataset as any).sizeMB > 0 && (
            <div className="flex items-center gap-1">
              <span className="font-mono text-foreground">{(dataset as any).sizeMB < 1 ? ((dataset as any).sizeMB).toFixed(2) : ((dataset as any).sizeMB).toFixed(1)}</span>
              <span>MB</span>
            </div>
          )}
          <div className="flex items-center gap-1">
            <Table className="h-3.5 w-3.5 shrink-0" />
            <span className="font-mono text-foreground">{dataset.rows >= 10000 ? Math.round(dataset.rows / 1000) + "K" : dataset.rows >= 1000 ? (dataset.rows % 1000 === 0 ? (dataset.rows / 1000) + "K" : (dataset.rows / 1000).toFixed(1) + "K") : dataset.rows.toLocaleString()}</span>
            <span>rows</span>
          </div>
          <div className="flex items-center gap-1">
            <Columns className="h-3.5 w-3.5 shrink-0" />
            <span className="font-mono text-foreground">{dataset.columns}</span>
            <span>cols</span>
          </div>
        </div>
        {(dataset as any).rateLimit && (
          <div className="mt-2 flex items-center gap-1.5 rounded-md bg-amber-500/10 px-2 py-1">
            <span className="text-xs text-amber-500">⚡ {(dataset as any).rateLimit}</span>
          </div>
        )}

        {/* Schema Preview */}
        <div className="mt-4 rounded-lg bg-muted/50 p-3">
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Schema Preview</p>
          <div className="flex flex-wrap gap-1.5">
            {dataset.schema.slice(0, 4).map((col, i) => (
              <span
                key={`${col.name}-${i}`}
                className="rounded bg-background px-1.5 py-0.5 font-mono text-xs text-muted-foreground border border-border max-w-[140px] truncate"
                title={col.name}
              >
                {col.name}
              </span>
            ))}
            {dataset.schema.length > 4 && (
              <span className="rounded bg-background px-1.5 py-0.5 text-xs text-muted-foreground border border-border">
                +{dataset.columns - 4} more
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
