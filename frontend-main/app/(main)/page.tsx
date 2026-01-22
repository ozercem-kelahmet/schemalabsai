import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, Database, Cpu, Clock, TrendingUp, Rocket } from "lucide-react"
import Link from "next/link"
import { mockModels, mockDatasets } from "@/lib/mock-data"
import { SourceBadge } from "@/components/datasets/source-badge"

export default function DashboardPage() {
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
          <p className="mt-2 text-sm text-muted-foreground">Build your first AI model in four simple steps</p>
          <div className="mt-4 grid gap-4 md:grid-cols-4">
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF]">
                1
              </div>
              <div>
                <p className="font-medium text-foreground">Connect Data</p>
                <p className="text-sm text-muted-foreground">Link multiple data sources</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF]">
                2
              </div>
              <div>
                <p className="font-medium text-foreground">Build Model</p>
                <p className="text-sm text-muted-foreground">Configure AI capabilities</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF]">
                3
              </div>
              <div>
                <p className="font-medium text-foreground">Evaluate & Chat</p>
                <p className="text-sm text-muted-foreground">Test in the playground</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0052CC]/20 font-mono text-sm font-bold text-[#2684FF]">
                4
              </div>
              <div>
                <p className="font-medium text-foreground">Deploy</p>
                <p className="text-sm text-muted-foreground">Ship to production via API</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="border-border bg-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Models Created</CardTitle>
            <Cpu className="h-4 w-4 text-[#2684FF]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">{mockModels.length}</div>
            <p className="text-xs text-muted-foreground">+1 from last week</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Datasets Available</CardTitle>
            <Database className="h-4 w-4 text-[#2684FF]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">{mockDatasets.length}</div>
            <p className="text-xs text-muted-foreground">Across 4 sources</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Training Hours</CardTitle>
            <Clock className="h-4 w-4 text-[#2684FF]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">4.2h</div>
            <p className="text-xs text-muted-foreground">Total compute time</p>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Avg. Accuracy</CardTitle>
            <TrendingUp className="h-4 w-4 text-[#2684FF]" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-foreground">90.7%</div>
            <p className="text-xs text-emerald-400">+2.3% improvement</p>
          </CardContent>
        </Card>
      </div>

      <div>
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-foreground">Recent Models</h3>
          <Link href="/models" className="text-sm text-[#2684FF] hover:text-[#0052CC]">
            View all
          </Link>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {mockModels.map((model) => (
            <Card key={model.id} className="border-border bg-card">
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="font-medium text-foreground">{model.name}</h4>
                    <p className="mt-1 text-sm text-muted-foreground">{model.description}</p>
                  </div>
                </div>
                <div className="mt-4 flex items-center gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Accuracy</p>
                    <p className="font-mono text-lg font-semibold text-emerald-400">
                      {(model.accuracy * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Data Sources</p>
                    <div className="mt-1 flex gap-1">
                      {model.datasets.map((ds) => (
                        <SourceBadge key={ds.datasetId} source={ds.source} size="sm" />
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Sync Mode</p>
                    <p className="text-sm capitalize text-muted-foreground">{model.syncMode}</p>
                  </div>
                </div>
                <div className="mt-4 flex gap-2">
                  <Link href={`/playground?model=${model.id}`} className="flex-1">
                    <Button variant="outline" size="sm" className="w-full bg-transparent">
                      Open in Playground
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
