"use client"

import type React from "react"

import { Checkbox } from "@/components/ui/checkbox"
import { FileSpreadsheet, Database, Server, Globe } from "lucide-react"
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
  availableSources?: DataSource[]
  availableVerticals?: string[]
  availableComplexity?: Complexity[]
  availableRowCount?: RowCount[]
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
  {
    id: "excel",
    label: "Excel",
    icon: <FileSpreadsheet className="h-4 w-4 text-[#217346]" />,
  },
  {
    id: "upload",
    label: "Uploaded Files",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#10B981]" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
      </svg>
    ),
  },
  {
    id: "generated",
    label: "Generated Data",
    icon: (
      <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#8B5CF6]" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    ),
  },
  {
    id: "postgresql",
    label: "PostgreSQL",
    icon: <Database className="h-4 w-4 text-[#336791]" />,
  },
  {
    id: "mysql",
    label: "MySQL",
    icon: <Database className="h-4 w-4 text-[#4479A1]" />,
  },
  {
    id: "mongodb",
    label: "MongoDB",
    icon: <Database className="h-4 w-4 text-[#47A248]" />,
  },
  {
    id: "snowflake",
    label: "Snowflake",
    icon: <Database className="h-4 w-4 text-[#29B5E8]" />,
  },
  {
    id: "pinecone",
    label: "Pinecone",
    icon: <Server className="h-4 w-4 text-[#7B61FF]" />,
  },
  {
    id: "weaviate",
    label: "Weaviate",
    icon: <Server className="h-4 w-4 text-[#00C8A8]" />,
  },
  {
    id: "chroma",
    label: "Chroma",
    icon: <Server className="h-4 w-4 text-[#FFD700]" />,
  },
  {
    id: "lancedb",
    label: "LanceDB",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#3B82F6]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "graphql",
    label: "GraphQL",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#E10098]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "rest",
    label: "REST API",
    icon: <Globe className="h-4 w-4 text-[#6366F1]" />,
  },
  {
    id: "gcs",
    label: "Google Cloud Storage",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#4285F4]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
  },
  {
    id: "aws-s3",
    label: "AWS S3",
    icon: <svg viewBox="0 0 24 24" className="h-4 w-4 text-[#FF9900]" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>,
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
  availableSources,
  availableVerticals,
  availableComplexity,
  availableRowCount,
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
          {(availableSources ? sources.filter(s => availableSources.includes(s.id)) : sources).map((source) => (
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
      {availableVerticals && availableVerticals.length > 0 && (
        <div>
          <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Vertical</h4>
          <div className="space-y-2">
            {availableVerticals.map((vertical) => (
              <div key={vertical} className="flex items-center gap-3">
                <Checkbox
                  id={`vertical-${vertical}`}
                  checked={selectedVerticals.includes(vertical as Vertical)}
                  onCheckedChange={() => {
                    const newVerticals = selectedVerticals.includes(vertical as Vertical)
                      ? selectedVerticals.filter(v => v !== vertical)
                      : [...selectedVerticals, vertical as Vertical]
                    onVerticalChange(newVerticals)
                  }}
                  className="border-border data-[state=checked]:bg-[#0052CC] data-[state=checked]:border-[#0052CC]"
                />
                <Label htmlFor={`vertical-${vertical}`} className="cursor-pointer text-sm text-foreground capitalize">
                  {vertical.replace(/-/g, " ")}
                </Label>
              </div>
            ))}
          </div>
        </div>
      )}
      {/* Complexity Filter */}
      <div>
        <h4 className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">Complexity</h4>
        <div className="space-y-2">
          {(availableComplexity ? complexityOptions.filter(o => availableComplexity.includes(o.id)) : complexityOptions).map((option) => (
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
          {(availableRowCount ? rowCountOptions.filter(o => availableRowCount.includes(o.id)) : rowCountOptions).map((option) => (
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
