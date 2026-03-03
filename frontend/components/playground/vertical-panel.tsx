"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import CodeMirror from "@uiw/react-codemirror"
import { python } from "@codemirror/lang-python"
import { yaml } from "@codemirror/lang-yaml"
import { oneDark } from "@codemirror/theme-one-dark"
import { EditorView } from "@codemirror/view"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { X, Upload, Check, AlertTriangle, Trash2, Eye, Pencil, Plus, ChevronDown, ChevronRight, Settings } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

interface VTool { id: string; name: string; description: string; code: string; hook: string; enabled: boolean; version: number; validation_status: string; validation_error: string; execution_order: number }
interface VAgent { id: string; name: string; description: string; code: string; role: string; enabled: boolean; version: number; pipeline_order: number; runs_if: string; parallel_with: string; validation_status: string; validation_error: string }
interface VConfig { id: string; name: string; description: string; config_yaml: string; enabled: boolean; version: number; model_id: string; language_config?: string }

interface VerticalPanelProps { open: boolean; onClose: () => void; modelId: string; modelName: string }

export function VerticalPanel({ open, onClose, modelId, modelName }: VerticalPanelProps) {
  const [verticals, setVerticals] = useState<VConfig[]>([])
  const [toolsMap, setToolsMap] = useState<Record<string, VTool[]>>({})
  const [agentsMap, setAgentsMap] = useState<Record<string, VAgent[]>>({})
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [newName, setNewName] = useState("")
  const [newDesc, setNewDesc] = useState("")
  const [uploadType, setUploadType] = useState<"tool" | "agent" | "config" | null>(null)
  const [targetVerticalId, setTargetVerticalId] = useState("")
  const [editingId, setEditingId] = useState<string | null>(null)
  const [uploadName, setUploadName] = useState("")
  const [uploadDescription, setUploadDescription] = useState("")
  const [uploadCode, setUploadCode] = useState("")
  const [uploadHook, setUploadHook] = useState("post_inference")
  const [uploading, setUploading] = useState(false)
  const [validationResult, setValidationResult] = useState<{status: string, error: string, checks: string[]} | null>(null)
  const [validated, setValidated] = useState(false)
  const [validating, setValidating] = useState(false)
  const [viewingCode, setViewingCode] = useState<{name: string, code: string} | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { if (open && modelId) fetchAll() }, [open, modelId])

  const fetchAll = async () => {
    setLoading(true)
    try {
      const res = await fetch(`/api/vertical/configs?model_id=${modelId}`, { credentials: "include" })
      if (res.ok) {
        const configs: VConfig[] = await res.json() || []
        setVerticals(configs)
        const active = configs.find(c => c.enabled)
        if (active && !selectedId) setSelectedId(active.id)
        const tm: Record<string, VTool[]> = {}
        const am: Record<string, VAgent[]> = {}
        await Promise.all(configs.map(async v => {
          const [t, a] = await Promise.all([
            fetch(`/api/vertical/tools?vertical_id=${v.id}`, { credentials: "include" }),
            fetch(`/api/vertical/agents?vertical_id=${v.id}`, { credentials: "include" }),
          ])
          tm[v.id] = t.ok ? await t.json() || [] : []
          am[v.id] = a.ok ? await a.json() || [] : []
        }))
        setToolsMap(tm); setAgentsMap(am)
      }
    } catch (e) { console.error(e) } finally { setLoading(false) }
  }

  const createVertical = async () => {
    if (!newName) return
    await fetch("/api/vertical/configs/create", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ model_id: modelId, name: newName, description: newDesc, config_yaml: `name: "${newName}"` }) })
    toast.success("Created"); setCreateOpen(false); setNewName(""); setNewDesc(""); fetchAll()
  }

  const activateVertical = async (id: string) => {
    const v = verticals.find(x => x.id === id)
    if (v?.enabled) {
      await fetch("/api/vertical/configs/update", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ id, enabled: false }) })
    } else {
      await fetch("/api/vertical/configs/activate", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ id, model_id: modelId }) })
    }
    fetchAll()
  }

  const deleteVertical = async (id: string, name: string) => {
    await fetch("/api/vertical/configs/delete", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ id }) })
    toast.success(`${name} deleted`); fetchAll()
  }

  const deleteItem = async (type: "tool" | "agent", id: string, name: string) => {
    const ep = type === "tool" ? "/api/vertical/tools/delete" : "/api/vertical/agents/delete"
    await fetch(ep, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ id }) })
    toast.success(`${name} deleted`); fetchAll()
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f || !f.name.endsWith('.py')) return
    const r = new FileReader()
    r.onload = ev => { setUploadCode(ev.target?.result as string); if (!uploadName) setUploadName(f.name.replace('.py', '')) }
    r.readAsText(f)
  }

  const handleValidate = async () => {
    if (!uploadCode) return
    setValidating(true); setValidationResult(null); setValidated(false)
    try {
      const ep = uploadType === "config" ? "/api/vertical/configs/validate" : "/api/vertical/tools/validate"
      const body = uploadType === "config" ? { config_yaml: uploadCode } : { code: uploadCode, script_type: uploadType, hook: uploadHook }
      const res = await fetch(ep, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify(body) })
      const result = await res.json()
      setValidationResult(result)
      if (result.status === "passed") setValidated(true)
      const steps = VSTEPS[(uploadType || "tool") as keyof typeof VSTEPS] || VSTEPS.tool
      setTimeout(() => { if (result.status === "passed") toast.success("Validation passed"); else toast.error("Validation failed") }, steps.length * 400 + 500)
    } catch (e: any) { toast.error(e.message) } finally { setValidating(false) }
  }

  const handleSave = async () => {
    if (!uploadName || !uploadCode || !validated) return
    setUploading(true)
    try {
      if (uploadType === "config") {
        const ep = editingId ? "/api/vertical/configs/update" : "/api/vertical/configs/create"
        const body = editingId ? { id: editingId, name: uploadName, description: uploadDescription, config_yaml: uploadCode } : { model_id: modelId, name: uploadName, description: uploadDescription, config_yaml: uploadCode }
        await fetch(ep, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify(body) })
      } else if (editingId) {
        const ep = uploadType === "tool" ? "/api/vertical/tools/update" : "/api/vertical/agents/update"
        const body: any = { id: editingId, code: uploadCode }; if (uploadType === "tool") body.hook = uploadHook
        await fetch(ep, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify(body) })
      } else {
        const ep = uploadType === "tool" ? "/api/vertical/tools/upload" : "/api/vertical/agents/upload"
        const body: any = { model_id: modelId, vertical_id: targetVerticalId, name: uploadName, description: uploadDescription, code: uploadCode }
        if (uploadType === "tool") body.hook = uploadHook
        await fetch(ep, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify(body) })
      }
      toast.success("Saved"); resetUpload(); fetchAll()
    } catch (e: any) { toast.error(e.message) } finally { setUploading(false) }
  }

  const openEdit = (type: "tool" | "agent" | "config", item: any, vId?: string) => {
    setUploadType(type); setEditingId(item.id); setTargetVerticalId(vId || "")
    setUploadName(item.name); setUploadDescription(item.description || "")
    setUploadCode(type === "config" ? (item.config_yaml || "") : item.code)
    if (type === "tool") setUploadHook(item.hook)
    setValidated(false); setValidationResult(null)
  }

  const openUpload = (type: "tool" | "agent", vId: string) => {
    setUploadType(type); setTargetVerticalId(vId); setEditingId(null)
    setUploadName(""); setUploadDescription(""); setUploadCode(""); setUploadHook("post_inference")
    setValidated(false); setValidationResult(null)
  }

  const resetUpload = () => {
    setUploadType(null); setEditingId(null); setTargetVerticalId(""); setUploadName(""); setUploadDescription("")
    setUploadCode(""); setUploadHook("post_inference"); setValidationResult(null); setValidated(false)
  }

  if (!open) return null

  const Row = ({ label, tag, valid, onView, onEdit, onDelete }: { label: string; tag?: string; valid?: string; onView?: () => void; onEdit?: () => void; onDelete?: () => void }) => (
    <div className="group flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50 transition-colors">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-xs text-foreground truncate">{label}</span>
        {tag && <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{tag}</span>}
        {valid === "passed" && <Check className="h-3 w-3 text-emerald-500 flex-shrink-0" />}
        {valid === "failed" && <AlertTriangle className="h-3 w-3 text-red-500 flex-shrink-0" />}
      </div>
      <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        {onView && <Button variant="ghost" size="icon" className="h-6 w-6 text-muted-foreground hover:text-foreground" onClick={onView}><Eye className="h-3 w-3" /></Button>}
        {onEdit && <Button variant="ghost" size="icon" className="h-6 w-6 text-muted-foreground hover:text-foreground" onClick={onEdit}><Pencil className="h-3 w-3" /></Button>}
        {onDelete && <Button variant="ghost" size="icon" className="h-6 w-6 text-muted-foreground hover:text-red-500" onClick={onDelete}><Trash2 className="h-3 w-3" /></Button>}
      </div>
    </div>
  )

  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/50" onClick={onClose} />
      <div className="fixed right-0 top-0 z-50 h-full w-[440px] bg-card border-l border-border shadow-2xl overflow-y-auto">
        <div className="sticky top-0 z-10 bg-card border-b border-border p-4 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-foreground">Vertical AI Runtime</h2>
            <p className="text-xs text-muted-foreground">{modelName}</p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => setCreateOpen(true)}>
              <Plus className="h-3 w-3 mr-1" /> New Vertical
            </Button>
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onClose}><X className="h-4 w-4" /></Button>
          </div>
        </div>

        <div className="p-3 space-y-1">
          {loading ? <div className="py-12 text-center text-xs text-muted-foreground">Loading...</div> : verticals.length === 0 ? (
            <div className="py-12 text-center">
              <p className="text-xs text-muted-foreground">No verticals yet</p>
              <Button variant="outline" size="sm" className="mt-3 text-xs" onClick={() => setCreateOpen(true)}><Plus className="h-3 w-3 mr-1" /> Create Vertical</Button>
            </div>
          ) : verticals.map(v => {
            const tools = toolsMap[v.id] || []
            const agents = agentsMap[v.id] || []
            const isOpen = expanded[v.id]
            return (
              <div key={v.id} className={cn("rounded-lg border transition-all", selectedId === v.id ? "border-foreground/20 bg-muted/40" : "border-transparent")}>
                {/* Vertical header - click to select */}
                <div className={cn("group flex items-center justify-between px-3 py-2.5 cursor-pointer rounded-lg transition-colors", selectedId !== v.id && "hover:bg-muted/30")} onClick={() => setSelectedId(selectedId === v.id ? null : v.id)}>
                  <div className="flex items-center gap-2 min-w-0">
                    <span className={cn("text-sm truncate", selectedId === v.id ? "font-semibold text-foreground" : "text-muted-foreground")}>{v.name}</span>
                    {v.enabled && <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-500">Active</span>}
                    {(() => { const t = toolsMap[v.id] || []; const a = agentsMap[v.id] || []; const hasConfig = v.config_yaml && v.config_yaml !== `name: "${v.name}"`; const complete = hasConfig && t.length > 0 && a.length > 0; return !complete ? <span className="text-[9px] text-amber-500">incomplete</span> : null })()}
                  </div>
                  <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-muted-foreground" onClick={e => { e.stopPropagation(); openEdit("config", v, v.id) }}><Pencil className="h-3 w-3" /></Button>
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-muted-foreground hover:text-red-500" onClick={e => { e.stopPropagation(); deleteVertical(v.id, v.name) }}><Trash2 className="h-3 w-3" /></Button>
                  </div>
                </div>

                {selectedId === v.id && (
                  <div className="px-3 pb-2 space-y-0.5 border-t border-border/50 pt-1">
                    <div className="flex items-center justify-end pb-1">
                      <button onClick={() => {
                        if (!v.enabled) {
                          const t = toolsMap[v.id] || []; const a = agentsMap[v.id] || []
                          const missing: string[] = []
                          if (!v.config_yaml || v.config_yaml === `name: "${v.name}"`) missing.push("System Config")
                          if (t.length === 0) missing.push("Tool")
                          if (a.length === 0) missing.push("Agent")
                          if (missing.length > 0) { toast.error(`Complete all 3 first: ${missing.join(", ")}`); return }
                        }
                        activateVertical(v.id)
                      }} className={cn("text-[10px] font-medium px-2.5 py-1 rounded transition-colors", v.enabled ? "bg-emerald-500/10 text-emerald-500 hover:bg-red-500/10 hover:text-red-400" : "bg-muted text-muted-foreground hover:text-foreground")}>{v.enabled ? "Deactivate" : "Activate"}</button>
                    </div>
                    <Row label="System Config" tag={`v${v.version}`} onView={() => setViewingCode({ name: v.name, code: v.config_yaml })} onEdit={() => openEdit("config", v, v.id)} onDelete={() => { fetch("/api/vertical/configs/update", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ id: v.id, config_yaml: "" }) }).then(() => { toast.success("Config cleared"); fetchAll() }) }} />
                    {tools.map((t: any) => <Row key={t.id} label={`${t.name}.py`} tag={t.hook} valid={t.validation_status} onView={() => setViewingCode({ name: t.name, code: t.code })} onEdit={() => openEdit("tool", t, v.id)} onDelete={() => deleteItem("tool", t.id, t.name)} />)}
                    <button className="text-[10px] text-muted-foreground hover:text-foreground transition-colors py-1.5 px-2" onClick={() => openUpload("tool", v.id)}>+ Add Tool</button>
                    {agents.map((a: any) => <Row key={a.id} label={`${a.name}.py`} tag={a.role} valid={a.validation_status} onView={() => setViewingCode({ name: a.name, code: a.code })} onEdit={() => openEdit("agent", a, v.id)} onDelete={() => deleteItem("agent", a.id, a.name)} />)}
                    <button className="text-[10px] text-muted-foreground hover:text-foreground transition-colors py-1.5 px-2" onClick={() => openUpload("agent", v.id)}>+ Add Agent</button>

                    {/* Language Layer Toggle */}
                    <div className="mt-2 pt-2 border-t border-border/50">
                      <div className="flex items-center justify-between px-2 py-1.5">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] font-medium text-muted-foreground">Language Layer</span>
                          {(() => { try { const lc = JSON.parse(v.language_config || "{}"); return lc.enabled ? <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#2684FF]/10 text-[#2684FF]">ON</span> : null } catch { return null } })()}
                        </div>
                        <button onClick={async () => {
                          const current = (() => { try { return JSON.parse(v.language_config || "{}") } catch { return {} } })()
                          const newEnabled = !current.enabled
                          const newConfig = JSON.stringify({ ...current, enabled: newEnabled, provider: current.provider || { type: "openai", model: "gpt-4o" } })
                          await fetch("/api/vertical/configs/update", { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ id: v.id, language_config: newConfig }) })
                          setVerticals(prev => prev.map(vc => vc.id === v.id ? { ...vc, language_config: newConfig } : vc))
                        }} className={(() => { try { const lc = JSON.parse(v.language_config || "{}"); return lc.enabled ? "text-[10px] font-medium px-2.5 py-1 rounded bg-[#2684FF]/10 text-[#2684FF] hover:bg-red-500/10 hover:text-red-400 transition-colors" : "text-[10px] font-medium px-2.5 py-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors" } catch { return "text-[10px] font-medium px-2.5 py-1 rounded bg-muted text-muted-foreground hover:text-foreground transition-colors" } })()}>
                          {(() => { try { const lc = JSON.parse(v.language_config || "{}"); return lc.enabled ? "Disable" : "Enable" } catch { return "Enable" } })()}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Create Vertical */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[400px] z-[60]">
          <DialogHeader><DialogTitle>New Vertical</DialogTitle></DialogHeader>
          <div className="space-y-3">
            <div><Label className="text-xs">Name</Label><Input value={newName} onChange={e => setNewName(e.target.value)} className="border-border bg-background mt-1" placeholder="Finance Detector" /></div>
            <div><Label className="text-xs">Description</Label><Input value={newDesc} onChange={e => setNewDesc(e.target.value)} className="border-border bg-background mt-1" placeholder="Optional" /></div>
          </div>
          <DialogFooter><Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button><Button onClick={createVertical} disabled={!newName} className="bg-[#0052CC] text-white hover:bg-[#003D99]">Create</Button></DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Upload/Edit */}
      <Dialog open={uploadType !== null} onOpenChange={o => { if (!o) resetUpload() }}>
        <DialogContent className="border-border bg-card sm:max-w-[600px] z-[60]">
          <DialogHeader>
            <DialogTitle>{editingId ? "Edit" : "Upload"} {uploadType === "config" ? "System Config" : uploadType === "tool" ? "Tool" : "Agent"}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div><Label className="text-xs">Name</Label><Input value={uploadName} onChange={e => setUploadName(e.target.value)} className="border-border bg-background mt-1" /></div>
            <div><Label className="text-xs">Description</Label><Input value={uploadDescription} onChange={e => setUploadDescription(e.target.value)} className="border-border bg-background mt-1" /></div>
            {uploadType === "tool" && (
              <div><Label className="text-xs">Hook</Label>
                <Select value={uploadHook} onValueChange={v => { setUploadHook(v); setValidated(false); setValidationResult(null) }}>
                  <SelectTrigger className="border-border bg-background mt-1"><SelectValue /></SelectTrigger>
                  <SelectContent><SelectItem value="pre_inference">Pre-Inference</SelectItem><SelectItem value="post_inference">Post-Inference</SelectItem><SelectItem value="validator">Validator</SelectItem></SelectContent>
                </Select>
              </div>
            )}
            <div>
              <div className="flex items-center justify-between"><Label className="text-xs">{uploadType === "config" ? "Config" : "Code"}</Label>
                {uploadType !== "config" && <Button variant="outline" size="sm" className="h-6 text-[10px]" onClick={() => fileInputRef.current?.click()}><Upload className="h-3 w-3 mr-1" />.py</Button>}
                <input ref={fileInputRef} type="file" accept=".py" onChange={handleFileUpload} className="hidden" />
              </div>
              <CodeMirror
                value={uploadCode}
                height="220px"
                theme={oneDark}
                extensions={[uploadType === "config" ? yaml() : python(), EditorView.lineWrapping]}
                placeholder={uploadType === "config" ? "Write system instructions (plain text or YAML)\n\nExamples:\n  You are a financial risk analyst\n  confidence_threshold: 0.75\n  Flag data with confidence below 0.5" : "def run(data, schema_output, config):\n    return {}"}
                onChange={(val) => { setUploadCode(val); setValidated(false); setValidationResult(null) }}
                basicSetup={{
                  lineNumbers: true,
                  highlightActiveLineGutter: true,
                  highlightActiveLine: true,
                  foldGutter: true,
                  bracketMatching: true,
                  autocompletion: true,
                  indentOnInput: true,
                }}
                style={{
                  fontSize: 12,
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px",
                  overflow: "hidden",
                }}
              />
            </div>
            {(validating || validationResult) && <VSteps validating={validating} result={validationResult} scriptType={uploadType || "tool"} />}
          </div>
          <DialogFooter className="gap-2">
            <Button variant="outline" size="sm" onClick={resetUpload}>Cancel</Button>
            <Button variant="outline" size="sm" onClick={() => { setUploadCode(""); setValidated(false); setValidationResult(null) }}>Clear</Button>
            <Button variant="outline" size="sm" onClick={handleValidate} disabled={validating || !uploadCode}>
              {validating ? "Validating..." : validated ? <><Check className="h-3 w-3 mr-1 text-emerald-500" />Validated</> : "Validate"}
            </Button>
            <Button size="sm" onClick={handleSave} disabled={uploading || !uploadName || !uploadCode || !validated} className="bg-[#0052CC] text-white hover:bg-[#003D99]">{uploading ? "Saving..." : "Save"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* View Code */}
      <Dialog open={viewingCode !== null} onOpenChange={o => { if (!o) setViewingCode(null) }}>
        <DialogContent className="border-border bg-card sm:max-w-[700px] z-[60]">
          <DialogHeader><DialogTitle className="text-sm">{viewingCode?.name}</DialogTitle></DialogHeader>
          <CodeMirror
            value={viewingCode?.code || ""}
            height="400px"
            theme={oneDark}
            extensions={[viewingCode?.name?.includes(".py") || !viewingCode?.name ? python() : yaml(), EditorView.lineWrapping]}
            readOnly={true}
            editable={false}
            basicSetup={{
              lineNumbers: true,
              highlightActiveLine: false,
              foldGutter: true,
            }}
            style={{
              fontSize: 12,
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
              overflow: "hidden",
            }}
          />
          <DialogFooter><Button variant="outline" size="sm" onClick={() => setViewingCode(null)}>Close</Button></DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

const VSTEPS = {
  tool: ["Size check", "Syntax check", "Security scan", "Interface validation", "Dry-run"],
  agent: ["Size check", "Syntax check", "Security scan", "Interface validation", "Dry-run"],
  config: ["Not empty", "Size check", "YAML/JSON parse", "Required fields", "Security check"],
}

function VSteps({ validating, result, scriptType }: { validating: boolean; result: { status: string; error: string; checks: string[] } | null; scriptType: string }) {
  const [vis, setVis] = useState(0), [cur, setCur] = useState(0)
  const steps = VSTEPS[scriptType as keyof typeof VSTEPS] || VSTEPS.tool
  useEffect(() => { if (validating) { setVis(0); setCur(0); let s = 0; const i = setInterval(() => { s++; if (s <= steps.length) { setCur(s); setVis(s) } else clearInterval(i) }, 600); return () => clearInterval(i) } }, [validating, steps.length])
  useEffect(() => { if (result && !validating) { let s = vis; const i = setInterval(() => { s++; if (s <= steps.length) { setVis(s); setCur(s) } else clearInterval(i) }, 400); return () => clearInterval(i) } }, [result, validating])
  const st = (i: number) => { if (!result && validating) { if (i < cur - 1) return "p"; if (i === cur - 1) return "c"; return "w" } if (result) { const n = result.checks?.length || 0; if (i < n) return "p"; if (i === n && result.error) return "f"; return "w" } return "w" }
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-3 space-y-1.5">
      {steps.map((s, i) => { if (i >= vis && !result) return null; const x = st(i); return (
        <div key={i} className={cn("flex items-center gap-2 text-xs", i < vis ? "opacity-100" : "opacity-0")}>
          {x === "c" && <span className="animate-spin h-3 w-3 border-2 border-foreground/30 border-t-foreground rounded-full" />}
          {x === "p" && <Check className="h-3 w-3 text-emerald-500" />}
          {x === "f" && <AlertTriangle className="h-3 w-3 text-red-500" />}
          {x === "w" && <span className="h-3 w-3 rounded-full border border-border" />}
          <span className={cn(x === "p" && "text-muted-foreground", x === "f" && "text-red-400", x === "c" && "text-foreground", x === "w" && "text-muted-foreground/40")}>{s}</span>
        </div>
      )})}
      {result?.error && <div className="text-xs text-red-400 pt-1.5 border-t border-border mt-1.5">{result.error}</div>}
    </div>
  )
}
