"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Download, Key, RotateCw, Shield, XCircle, Eye, Plus, History } from "lucide-react"
import { toast } from "sonner"

interface Bundle {
  id: string
  model_id: string
  model_name: string
  model_version: number
  encrypted_size: number
  key_id: string
  ciphertext_sha256: string
  deployment_target: string
  status: string
  download_count: number
  last_downloaded_at: string
  revoked_at: string
  created_at: string
}

interface AuditEntry {
  id: string
  bundle_id: string
  action: string
  ip_address: string
  details: string
  created_at: string
}

export function DedicatedDeployments() {
  const [bundles, setBundles] = useState<Bundle[]>([])
  const [models, setModels] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [createOpen, setCreateOpen] = useState(false)
  const [selectedModelID, setSelectedModelID] = useState("")
  const [deploymentTarget, setDeploymentTarget] = useState("")
  const [keyDialogOpen, setKeyDialogOpen] = useState(false)
  const [keyReveal, setKeyReveal] = useState<any>(null)
  const [auditOpen, setAuditOpen] = useState(false)
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([])
  const [auditBundleID, setAuditBundleID] = useState("")

  const fetchBundles = async () => {
    try {
      const res = await fetch("/api/dedicated/bundles", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setBundles(Array.isArray(data) ? data : [])
      }
    } catch {}
    setLoading(false)
  }

  const fetchModels = async () => {
    try {
      const res = await fetch("/api/models", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setModels(Array.isArray(data) ? data : (data.models || []))
      }
    } catch {}
  }

  useEffect(() => {
    fetchBundles()
    fetchModels()
  }, [])

  const handleCreate = async () => {
    if (!selectedModelID) {
      toast.error("Select a model")
      return
    }
    try {
      const res = await fetch("/api/dedicated/bundles/create", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: selectedModelID, deployment_target: deploymentTarget }),
      })
      if (res.ok) {
        toast.success("Encrypted bundle created")
        setCreateOpen(false)
        setSelectedModelID("")
        setDeploymentTarget("")
        fetchBundles()
      } else {
        const err = await res.text()
        toast.error("Failed: " + err)
      }
    } catch {
      toast.error("Network error")
    }
  }

  const handleIssueToken = async (bundleID: string) => {
    try {
      const res = await fetch("/api/dedicated/bundles/token", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bundle_id: bundleID }),
      })
      if (res.ok) {
        const data = await res.json()
        const fullURL = window.location.origin + data.url
        toast.success("Download token issued (15 min)")
        window.open(fullURL, "_blank")
        fetchBundles()
      } else {
        toast.error(await res.text())
      }
    } catch {
      toast.error("Network error")
    }
  }

  const handleRevealKey = async (bundleID: string) => {
    try {
      const res = await fetch("/api/dedicated/bundles/key", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bundle_id: bundleID }),
      })
      if (res.ok) {
        const data = await res.json()
        setKeyReveal(data)
        setKeyDialogOpen(true)
      } else {
        toast.error(await res.text())
      }
    } catch {
      toast.error("Network error")
    }
  }

  const handleRotate = async (bundleID: string) => {
    if (!confirm("Rotate encryption key? Old key becomes invalid immediately.")) return
    try {
      const res = await fetch("/api/dedicated/bundles/rotate", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bundle_id: bundleID }),
      })
      if (res.ok) {
        toast.success("Key rotated. Redistribute new key to customer.")
        fetchBundles()
      } else {
        toast.error(await res.text())
      }
    } catch {
      toast.error("Network error")
    }
  }

  const handleRevoke = async (bundleID: string) => {
    if (!confirm("Revoke this bundle? This action cannot be undone.")) return
    try {
      const res = await fetch("/api/dedicated/bundles/revoke", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bundle_id: bundleID }),
      })
      if (res.ok) {
        toast.success("Bundle revoked")
        fetchBundles()
      } else {
        toast.error(await res.text())
      }
    } catch {
      toast.error("Network error")
    }
  }

  const handleAudit = async (bundleID: string) => {
    setAuditBundleID(bundleID)
    try {
      const res = await fetch(`/api/dedicated/audit?bundle_id=${bundleID}`, { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setAuditEntries(Array.isArray(data) ? data : [])
        setAuditOpen(true)
      }
    } catch {
      toast.error("Failed to load audit log")
    }
  }

  const fmtSize = (b: number) => {
    if (b < 1024) return b + " B"
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB"
    if (b < 1024 * 1024 * 1024) return (b / 1024 / 1024).toFixed(1) + " MB"
    return (b / 1024 / 1024 / 1024).toFixed(2) + " GB"
  }

  return (
    <div className="rounded-xl border border-border bg-card p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <Shield className="h-5 w-5 text-[#0052CC]" />
            Dedicated Deployments
          </h3>
          <p className="text-sm text-muted-foreground mt-0.5">
            AES-256 encrypted checkpoint bundles for on-premises or air-gapped deployment.
          </p>
        </div>
        <Button className="bg-[#0052CC] hover:bg-[#003D99] text-white" onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4 mr-1.5" />
          Create Bundle
        </Button>
      </div>

      {loading ? (
        <div className="text-sm text-muted-foreground py-6 text-center">Loading...</div>
      ) : bundles.length === 0 ? (
        <div className="text-sm text-muted-foreground py-8 text-center border border-dashed border-border rounded-lg">
          No dedicated bundles yet. Create one to export an encrypted checkpoint.
        </div>
      ) : (
        <div className="space-y-3">
          {bundles.map((b) => {
            const isRevoked = b.revoked_at && b.revoked_at !== "0001-01-01T00:00:00Z"
            return (
              <div key={b.id} className="rounded-lg border border-border p-4 bg-muted/20">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-foreground">{b.model_name} <span className="text-muted-foreground">v{b.model_version}</span></span>
                      {isRevoked ? (
                        <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-[10px] font-medium text-red-500">Revoked</span>
                      ) : (
                        <span className="rounded-full bg-[#36B37E]/15 px-2 py-0.5 text-[10px] font-medium text-[#36B37E]">{b.status}</span>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1 font-mono">{b.id}</div>
                  </div>
                  <div className="text-right text-xs text-muted-foreground">
                    <div>{fmtSize(b.encrypted_size)}</div>
                    <div>{new Date(b.created_at).toLocaleDateString()}</div>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-muted-foreground mb-3">
                  <div><span className="font-medium text-foreground">Key ID:</span> <span className="font-mono">{b.key_id.slice(0, 16)}...</span></div>
                  <div><span className="font-medium text-foreground">Target:</span> {b.deployment_target || "-"}</div>
                  <div><span className="font-medium text-foreground">Downloads:</span> {b.download_count}</div>
                  <div><span className="font-medium text-foreground">SHA-256:</span> <span className="font-mono">{b.ciphertext_sha256.slice(0, 12)}...</span></div>
                </div>
                {!isRevoked && (
                  <div className="flex flex-wrap gap-2">
                    <Button size="sm" variant="outline" onClick={() => handleIssueToken(b.id)}>
                      <Download className="h-3.5 w-3.5 mr-1.5" />
                      Download
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => handleRevealKey(b.id)}>
                      <Eye className="h-3.5 w-3.5 mr-1.5" />
                      Reveal Key
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => handleRotate(b.id)}>
                      <RotateCw className="h-3.5 w-3.5 mr-1.5" />
                      Rotate Key
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => handleAudit(b.id)}>
                      <History className="h-3.5 w-3.5 mr-1.5" />
                      Audit Log
                    </Button>
                    <Button size="sm" variant="outline" className="text-red-500 hover:text-red-600" onClick={() => handleRevoke(b.id)}>
                      <XCircle className="h-3.5 w-3.5 mr-1.5" />
                      Revoke
                    </Button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="border-border bg-card">
          <DialogHeader>
            <DialogTitle className="text-foreground">Create Encrypted Bundle</DialogTitle>
            <DialogDescription>Export a fine-tuned model as an AES-256 encrypted artifact for on-prem deployment.</DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-4">
            <div>
              <Label className="text-xs text-muted-foreground mb-1.5 block">Model</Label>
              <select
                className="w-full h-9 rounded-md border border-border bg-background px-3 text-sm"
                value={selectedModelID}
                onChange={(e) => setSelectedModelID(e.target.value)}
              >
                <option value="">Select a fine-tuned model...</option>
                {models.map((m: any) => (
                  <option key={m.id} value={m.id}>{m.name} (v{m.version || 1})</option>
                ))}
              </select>
            </div>
            <div>
              <Label className="text-xs text-muted-foreground mb-1.5 block">Deployment Target (optional)</Label>
              <Input
                placeholder="e.g. customer on-prem A100 cluster"
                className="border-border bg-background"
                value={deploymentTarget}
                onChange={(e) => setDeploymentTarget(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button className="bg-[#0052CC] hover:bg-[#003D99] text-white" onClick={handleCreate}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={keyDialogOpen} onOpenChange={setKeyDialogOpen}>
        <DialogContent className="border-border bg-card max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-foreground flex items-center gap-2">
              <Key className="h-5 w-5 text-amber-500" />
              Decryption Key Revealed
            </DialogTitle>
            <DialogDescription className="text-amber-500">
              Transmit this key only over encrypted out-of-band channel. Never commit to repository.
            </DialogDescription>
          </DialogHeader>
          {keyReveal && (
            <div className="space-y-3 py-4">
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5 block">Key (hex, 256-bit)</Label>
                <div className="font-mono text-xs bg-muted p-3 rounded border border-border break-all">{keyReveal.key_hex}</div>
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-2"
                  onClick={() => {
                    navigator.clipboard.writeText(keyReveal.key_hex)
                    toast.success("Key copied")
                  }}
                >
                  Copy Key
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div><span className="text-muted-foreground">Algorithm:</span> <span className="font-mono">{keyReveal.algorithm}</span></div>
                <div><span className="text-muted-foreground">Fingerprint:</span> <span className="font-mono">{keyReveal.fingerprint}</span></div>
              </div>
              <p className="text-xs text-muted-foreground">{keyReveal.warning}</p>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => { setKeyDialogOpen(false); setKeyReveal(null) }}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={auditOpen} onOpenChange={setAuditOpen}>
        <DialogContent className="border-border bg-card max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="text-foreground">Audit Log</DialogTitle>
            <DialogDescription>Bundle: <span className="font-mono text-xs">{auditBundleID}</span></DialogDescription>
          </DialogHeader>
          <div className="overflow-y-auto flex-1 py-2">
            {auditEntries.length === 0 ? (
              <div className="text-sm text-muted-foreground py-6 text-center">No audit entries.</div>
            ) : (
              <div className="space-y-2">
                {auditEntries.map((e) => (
                  <div key={e.id} className="text-xs border border-border rounded p-2 bg-muted/20">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-foreground">{e.action}</span>
                      <span className="text-muted-foreground">{new Date(e.created_at).toLocaleString()}</span>
                    </div>
                    <div className="text-muted-foreground mt-1">
                      <span className="font-mono">{e.ip_address}</span>
                      {e.details && <span className="ml-2">{e.details}</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAuditOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
