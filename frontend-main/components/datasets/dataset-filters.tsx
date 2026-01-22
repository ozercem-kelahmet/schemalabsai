"use client"

import type React from "react"

import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import type { DataSource, Vertical, Complexity, RowCount } from "@/lib/types"

interface DatasetFiltersProps {
  selectedSources: DataSource[]
  selectedVerticals: Vertical[]
  selectedComplexity: Complexity[]
  selectedRowCount: RowCount[]
  onSourceChange: (sources: DataSource[]) => void
  onVerticalChange: (verticals: Vertical[]) => void
  onComplexityChange: (complexity: Complexity[]) => void
  onRowCountChange: (rowCount: RowCount[]) => void
}

const sources: { id: DataSource; label: string; icon: React.ReactNode }[] = [
  {
    id: "databricks",
    label: "Databricks",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#FF3621]" fill="currentColor">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    id: "supabase",
    label: "Supabase",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#3ECF8E]" fill="currentColor">
        <path d="M21.362 9.354H12V.396a.396.396 0 00-.716-.233L2.203 12.424l-.401.562a1.04 1.04 0 00.836 1.659H12v8.959a.396.396 0 00.716.233l9.081-12.261.401-.562a1.04 1.04 0 00-.836-1.66z" />
      </svg>
    ),
  },
  {
    id: "api",
    label: "API",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#2684FF]" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 6h16M4 12h16M4 18h16" />
      </svg>
    ),
  },
  {
    id: "google-drive",
    label: "Google Drive",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
        <path d="M12 2L4 14h4l4-7 4 7h4L12 2z" fill="#4285F4" />
        <path d="M4 14l4 8h8l4-8H4z" fill="#FBBC04" />
      </svg>
    ),
  },
]

const verticals: { id: Vertical; label: string }[] = [
  { id: "finance", label: "Finance" },
  { id: "healthcare", label: "Healthcare" },
  { id: "e-commerce", label: "E-commerce" },
  { id: "marketing", label: "Marketing" },
  { id: "hr", label: "HR" },
  { id: "operations", label: "Operations" },
]

const complexityOptions: { id: Complexity; label: string; description: string }[] = [
  { id: "simple", label: "Simple", description: "5-10 columns" },
  { id: "medium", label: "Medium", description: "10-25 columns" },
  { id: "advanced", label: "Advanced", description: "25+ columns" },
]

const rowCountOptions: { id: RowCount; label: string; description: string }[] = [
  { id: "small", label: "Small", description: "< 1K rows" },
  { id: "medium", label: "Medium", description: "1K-10K rows" },
  { id: "large", label: "Large", description: "10K+ rows" },
]

export function DatasetFilters({
  selectedSources,
  selectedVerticals,
  selectedComplexity,
  selectedRowCount,
  onSourceChange,
  onVerticalChange,
  onComplexityChange,
  onRowCountChange,
}: DatasetFiltersProps) {
  const toggleFilter = <T extends string>(current: T[], value: T, onChange: (values: T[]) => void) => {
    if (current.includes(value)) {
      onChange(current.filter((v) => v !== value))
    } else {
      onChange([...current, value])
    }
  }

  return (
    <div className="w-72 shrink-0 space-y-6 rounded-xl border border-border bg-card p-5">
      <div>
        <h3 className="mb-3 text-sm font-semibold text-foreground">Filters</h3>
        <p className="text-xs text-muted-foreground">Narrow down datasets</p>
      </div>

      {/* Source Filter */}
      <div>
        <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Source</h4>
        <div className="space-y-2">
          {sources.map((source) => (
            <div key={source.id} className="flex items-center gap-3">
              <Checkbox
                id={`source-${source.id}`}
                checked={selectedSources.includes(source.id)}
                onCheckedChange={() => toggleFilter(selectedSources, source.id, onSourceChange)}
                className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
              />
              <Label
                htmlFor={`source-${source.id}`}
                className="flex cursor-pointer items-center gap-2 text-sm text-foreground"
              >
                {source.icon}
                {source.label}
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Vertical Filter */}
      <div>
        <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Vertical</h4>
        <div className="space-y-2">
          {verticals.map((vertical) => (
            <div key={vertical.id} className="flex items-center gap-3">
              <Checkbox
                id={`vertical-${vertical.id}`}
                checked={selectedVerticals.includes(vertical.id)}
                onCheckedChange={() => toggleFilter(selectedVerticals, vertical.id, onVerticalChange)}
                className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
              />
              <Label htmlFor={`vertical-${vertical.id}`} className="cursor-pointer text-sm text-foreground">
                {vertical.label}
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Complexity Filter */}
      <div>
        <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Complexity</h4>
        <div className="space-y-2">
          {complexityOptions.map((option) => (
            <div key={option.id} className="flex items-center gap-3">
              <Checkbox
                id={`complexity-${option.id}`}
                checked={selectedComplexity.includes(option.id)}
                onCheckedChange={() => toggleFilter(selectedComplexity, option.id, onComplexityChange)}
                className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
              />
              <Label htmlFor={`complexity-${option.id}`} className="cursor-pointer text-sm text-foreground">
                <span>{option.label}</span>
                <span className="ml-1 text-xs text-muted-foreground">({option.description})</span>
              </Label>
            </div>
          ))}
        </div>
      </div>

      {/* Row Count Filter */}
      <div>
        <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Row Count</h4>
        <div className="space-y-2">
          {rowCountOptions.map((option) => (
            <div key={option.id} className="flex items-center gap-3">
              <Checkbox
                id={`rows-${option.id}`}
                checked={selectedRowCount.includes(option.id)}
                onCheckedChange={() => toggleFilter(selectedRowCount, option.id, onRowCountChange)}
                className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
              />
              <Label htmlFor={`rows-${option.id}`} className="cursor-pointer text-sm text-foreground">
                <span>{option.label}</span>
                <span className="ml-1 text-xs text-muted-foreground">({option.description})</span>
              </Label>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
