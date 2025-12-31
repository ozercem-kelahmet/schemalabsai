"use client"

import { useState, useEffect } from "react"
import { Sidebar } from "@/components/sidebar"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"

const API_BASE = "/api"

export default function UsagePage() {
  const [stats, setStats] = useState({
    totalModels: 0,
    totalRequests: 0,
    totalKeys: 0,
    totalFiles: 0,
    totalStorage: 0
  })
  const [keys, setKeys] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [keysRes, modelsRes, filesRes] = await Promise.all([
          fetch(API_BASE + "/keys", { credentials: "include" }),
          fetch(API_BASE + "/models/finetuned", { credentials: "include" }),
          fetch(API_BASE + "/files", { credentials: "include" })
        ])
        
        const keysData = await keysRes.json()
        const modelsData = await modelsRes.json()
        const filesData = await filesRes.json()
        
        let totalRequests = 0
        if (Array.isArray(keysData)) {
          totalRequests = keysData.reduce((acc, k) => acc + (k.requests || 0), 0)
          setKeys(keysData)
        }
        
        const modelCount = Array.isArray(modelsData) ? modelsData.length : (modelsData?.models?.length || 0)
        const fileCount = Array.isArray(filesData) ? filesData.length : 0
        const storageUsed = Array.isArray(filesData) 
          ? filesData.reduce((acc, f) => acc + (f.size || 0), 0) / (1024 * 1024 * 1024) 
          : 0
        
        setStats({
          totalModels: modelCount,
          totalRequests,
          totalKeys: Array.isArray(keysData) ? keysData.length : 0,
          totalFiles: fileCount,
          totalStorage: parseFloat(storageUsed.toFixed(2))
        })
      } catch (e) { 
        console.error("Fetch error:", e) 
      }
      finally { setLoading(false) }
    }
    fetchData()
  }, [])

  const planLimits = [
    { name: "Current session", used: 31, limit: 100, resetText: "Resets in 10 min" },
    { name: "All models", used: stats.totalRequests, limit: 50000, resetText: "Resets Fri 11:59 PM" },
  ]

  const resourceLimits = [
    { name: "Fine-tuned Models", used: stats.totalModels, limit: 100, unit: "models" },
    { name: "API Keys", used: stats.totalKeys, limit: 10, unit: "keys" },
    { name: "File Uploads", used: stats.totalFiles, limit: 500, unit: "files" },
    { name: "Storage", used: stats.totalStorage, limit: 50, unit: "GB" },
  ]

  return (
    <Sidebar>
      <div className="flex-1 overflow-auto bg-background">
        <div className="max-w-3xl mx-auto p-8">
          <h1 className="text-2xl font-semibold mb-8">Usage</h1>
          
          {loading ? <p className="text-muted-foreground">Loading...</p> : (
            <div className="space-y-8">
              <div>
                <h2 className="text-base font-medium mb-1">Plan usage limits</h2>
                <div className="space-y-6 mt-4">
                  {planLimits.map((item) => {
                    const pct = item.limit > 0 ? (item.used / item.limit) * 100 : 0
                    return (
                      <div key={item.name}>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm">{item.name}</span>
                          <span className="text-sm text-muted-foreground">{Math.round(pct)}% used</span>
                        </div>
                        <Progress value={Math.min(pct, 100)} className="h-2" />
                        <p className="text-xs text-muted-foreground mt-1">{item.resetText}</p>
                      </div>
                    )
                  })}
                </div>
              </div>

              <div>
                <h2 className="text-base font-medium mb-1">Weekly limits</h2>
                <p className="text-sm text-muted-foreground mb-4">Learn more about usage limits</p>
                <div className="space-y-6">
                  {resourceLimits.map((item) => {
                    const pct = item.limit > 0 ? (item.used / item.limit) * 100 : 0
                    return (
                      <div key={item.name}>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm">{item.name}</span>
                          <span className="text-sm text-muted-foreground">{Math.round(pct)}% used</span>
                        </div>
                        <Progress value={Math.min(pct, 100)} className="h-2" />
                        <p className="text-xs text-muted-foreground mt-1">{item.used} / {item.limit} {item.unit}</p>
                      </div>
                    )
                  })}
                </div>
              </div>

              <div>
                <h2 className="text-base font-medium mb-1">Extra usage</h2>
                <p className="text-sm text-muted-foreground mb-4">Turn on extra usage to keep using SchemaLabs if you hit a limit.</p>
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <p className="text-sm font-medium">$0.00 spent</p>
                    <p className="text-xs text-muted-foreground">Resets Jan 1</p>
                  </div>
                  <Button variant="outline" size="sm">Buy extra usage</Button>
                </div>
              </div>

              {keys.length > 0 && (
                <div>
                  <h2 className="text-base font-medium mb-4">API Key Usage</h2>
                  <div className="border rounded-lg overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-muted/50">
                        <tr>
                          <th className="text-left py-2 px-4 font-medium">Key</th>
                          <th className="text-right py-2 px-4 font-medium">Requests</th>
                        </tr>
                      </thead>
                      <tbody>
                        {keys.map((k, i) => (
                          <tr key={i} className="border-t">
                            <td className="py-2 px-4">{k.name || "Key " + (i + 1)}</td>
                            <td className="py-2 px-4 text-right">{(k.requests || 0).toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Sidebar>
  )
}
