"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, Database, Cpu, Clock, TrendingUp, Rocket } from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"

interface Model {
  id: string
  name: string
  accuracy: number
  epochs: number
  created_at: string
  source_name?: string
  connection_names?: string
  source_file_names?: string
  training_duration?: number
}

interface DatasetFile {
  id: string
  filename: string
  source: string
}

export default function DashboardPage() {
  const router = useRouter()
  const [models, setModels] = useState<Model[]>([])
  const [datasets, setDatasets] = useState<DatasetFile[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [modelsRes, datasetsRes, connectionsRes] = await Promise.all([
          fetch("/api/models/finetuned", { credentials: "include" }),
          fetch("/api/files", { credentials: "include" }),
          fetch("/api/connections", { credentials: "include" }).catch(() => ({ ok: false }))
        ])
        if (modelsRes.ok) {
          // @ts-ignore
          const data = await modelsRes.json()
          setModels(data.models || [])
        }
        if (datasetsRes.ok) {
          // @ts-ignore
          const filesData = await datasetsRes.json()
          const files = (filesData.files || []).filter((f: any) => !f.is_merged && !f.filename?.includes("_merged_all"))
          
          let allDatasets = [...files]
          
          if (connectionsRes.ok) {
            // @ts-ignore
            const connectionsData = await connectionsRes.json()
            const connections = connectionsData.connections || []
            // Count selected tables per connection, not just 1 per connection
            for (const conn of connections) {
              let selectedTables: string[] = []
              try {
                if (conn.selected_tables) selectedTables = JSON.parse(conn.selected_tables)
              } catch {}
              if (selectedTables.length > 0) {
                for (const t of selectedTables) {
                  allDatasets.push({ ...conn, id: conn.id + "::" + t, name: conn.name + " - " + t })
                }
              } else {
                allDatasets.push(conn)
              }
            }
          }
          
          setDatasets(allDatasets)
        }
      } catch (e) {
        console.error("Failed to fetch data:", e)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const avgAccuracy = models.length > 0 
    ? models.reduce((sum, m) => {
        const acc = m.accuracy || 0
        return sum + (acc > 1 ? acc : acc * 100)
      }, 0) / models.length 
    : 0
  
  const totalTrainingSeconds = models.reduce((sum, m) => {
    if (m.training_duration && m.training_duration > 0) {
      return sum + m.training_duration
    }
    // Estimate: ~2 seconds per epoch for models without duration
    const epochs = m.epochs || 5
    return sum + (epochs * 2)
  }, 0)
  const trainingHours = (totalTrainingSeconds / 3600).toFixed(1)
  const recentModels = models.slice(0, 2)

  const handleModelClick = async (model: Model) => {
    try {
      const res = await fetch("/api/queries?model_id=" + model.id, { credentials: "include" })
      if (res.ok) {
        // @ts-ignore
        const data = await res.json()
        if (data.queries && data.queries.length > 0) {
          router.push("/playground/" + data.queries[0].id)
          return
        }
      }
    } catch (e) {
      console.error("Failed to get query:", e)
    }
    router.push("/playground?model=" + model.id)
  }

  const getSourceNames = (model: Model): string[] => {
    if (model.connection_names) return model.connection_names.split(",").map(s => s.trim()).filter(Boolean)
    if (model.source_file_names) return model.source_file_names.split(",").map(s => s.trim()).filter(n => n && n !== "0 files merged")
    if (model.source_name && model.source_name !== "0 files merged") return [model.source_name]
    return []
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Welcome back</h2>
          <p className="mt-1 text-muted-foreground">Build AI models directly on your tabular data</p>
        </div>
        <Link href="/build">
          <Button className="gap-2 bg-[#0052CC] text-white hover:bg-[#003D99]">
            <Rocket className="h-4 w-4" />
            Build New Model
            <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>

      <Card className="border-[#0052CC]/20 bg-gradient-to-br from-[#0052CC]/10 to-transparent">
        <CardContent className="p-6">
          <h3 className="text-lg font-semibold text-foreground">Getting Started with Schema</h3>
          <p className="mt-2 text-sm text-muted-foreground">Build your first AI model in five simple steps</p>
          <div className="mt-4 grid gap-4 md:grid-cols-5">
            <Link href="/datasets" className="flex gap-3 group cursor-pointer">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF] group-hover:bg-[#0052CC]/30 transition-colors">1</div>
              <div><p className="font-medium text-foreground group-hover:text-[#2684FF] transition-colors">Connect Data</p><p className="text-sm text-muted-foreground">Link multiple data sources</p></div>
            </Link>
            <Link href="/build" className="flex gap-3 group cursor-pointer">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF] group-hover:bg-[#0052CC]/30 transition-colors">2</div>
              <div><p className="font-medium text-foreground group-hover:text-[#2684FF] transition-colors">Build Model</p><p className="text-sm text-muted-foreground">Configure AI capabilities</p></div>
            </Link>
            <Link href="/models" className="flex gap-3 group cursor-pointer">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF] group-hover:bg-[#0052CC]/30 transition-colors">3</div>
              <div><p className="font-medium text-foreground group-hover:text-[#2684FF] transition-colors">Enhance Model</p><p className="text-sm text-muted-foreground">Add agents & tools</p></div>
            </Link>
            <Link href="/playground" className="flex gap-3 group cursor-pointer">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF] group-hover:bg-[#0052CC]/30 transition-colors">4</div>
              <div><p className="font-medium text-foreground group-hover:text-[#2684FF] transition-colors">Evaluate & Chat</p><p className="text-sm text-muted-foreground">Test in playground</p></div>
            </Link>
            <Link href="/configuration" className="flex gap-3 group cursor-pointer">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF] group-hover:bg-[#0052CC]/30 transition-colors">5</div>
              <div><p className="font-medium text-foreground group-hover:text-[#2684FF] transition-colors">Deploy</p><p className="text-sm text-muted-foreground">Ship to production</p></div>
            </Link>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="border-border bg-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Models Created</CardTitle>
            <Cpu className="h-4 w-4 text-[#2684FF]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">{loading ? "..." : models.length}</div>
            <p className="text-xs text-muted-foreground">Fine-tuned models</p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Datasets Available</CardTitle>
            <Database className="h-4 w-4 text-[#2684FF]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">{loading ? "..." : datasets.length}</div>
            <p className="text-xs text-muted-foreground">Uploaded files</p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Training Hours</CardTitle>
            <Clock className="h-4 w-4 text-[#2684FF]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">{loading ? "..." : trainingHours + "h"}</div>
            <p className="text-xs text-muted-foreground">Total compute time</p>
          </CardContent>
        </Card>
        <Card className="border-border bg-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Avg. Accuracy</CardTitle>
            <TrendingUp className="h-4 w-4 text-[#2684FF]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">{loading ? "..." : avgAccuracy.toFixed(1) + "%"}</div>
            <p className="text-xs text-emerald-400">Across all models</p>
          </CardContent>
        </Card>
      </div>

      <div>
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-foreground">Recent Models</h3>
          <Link href="/models" className="text-sm text-[#2684FF] hover:text-[#0052CC]">View all</Link>
        </div>
        {loading ? (
          <div className="text-muted-foreground">Loading...</div>
        ) : recentModels.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2">
            {recentModels.map((model) => {
              const sources = getSourceNames(model)
              return (
                <Card key={model.id} className="border-border bg-card">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="font-medium text-foreground">{model.name}</h4>
                        <p className="mt-1 text-sm text-muted-foreground">Fine-tuned model</p>
                      </div>
                    </div>
                    <div className="mt-4 flex items-center gap-4">
                      <div>
                        <p className="text-xs text-muted-foreground">Accuracy</p>
                        <p className="font-mono text-lg font-semibold text-emerald-400">{(model.accuracy || 0).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Data Sources</p>
                        <div className="mt-1 flex gap-1">
                          {sources.slice(0, 2).map((name, i) => (
                            <span key={i} className="inline-flex items-center gap-1 rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-600 dark:text-emerald-400">
                              <Database className="h-3 w-3" />
                              {name.length > 12 ? name.slice(0, 12) + "..." : name}
                            </span>
                          ))}
                          {sources.length > 2 && <span className="text-xs text-muted-foreground">+{sources.length - 2}</span>}
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Sync Mode</p>
                        <p className="text-sm capitalize text-muted-foreground">Real-Time</p>
                      </div>
                    </div>
                    <div className="mt-4 flex gap-2">
                      <Button variant="outline" size="sm" className="w-full bg-transparent" onClick={() => handleModelClick(model)}>
                        Open in Playground
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        ) : (
          <div className="text-muted-foreground">No models yet. Build your first model!</div>
        )}
      </div>
    </div>
  )
}
