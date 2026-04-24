"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Plus, Gift, Copy, AlertCircle } from "lucide-react"
import { toast } from "sonner"

interface GiftCodeRow {
  code: string
  provider: string
  total_credits: number
  used_credits: number
  valid_until: string
  redeemed_by: string
  redeemed_at: string
  created_at: string
}

export default function AdminGiftCodesPage() {
  const [codes, setCodes] = useState<GiftCodeRow[]>([])
  const [loading, setLoading] = useState(true)
  const [forbidden, setForbidden] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [code, setCode] = useState("")
  const [provider, setProvider] = useState("manual")
  const [totalCredits, setTotalCredits] = useState("50")
  const [validUntil, setValidUntil] = useState("")

  const fetchCodes = async () => {
    try {
      const res = await fetch("/api/admin/gift-codes", { credentials: "include" })
      if (res.status === 403) {
        setForbidden(true)
        setLoading(false)
        return
      }
      if (res.ok) {
        const data = await res.json()
        setCodes(Array.isArray(data) ? data : [])
      }
    } catch {}
    setLoading(false)
  }

  useEffect(() => {
    fetchCodes()
  }, [])

  const handleCreate = async () => {
    const credits = parseFloat(totalCredits)
    if (!code.trim() || !provider || !credits || credits <= 0) {
      toast.error("Fill all required fields")
      return
    }
    try {
      const res = await fetch("/api/admin/gift-codes/create", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: code.trim(),
          provider,
          total_credits: credits,
          valid_until: validUntil,
        }),
      })
      if (res.ok) {
        toast.success("Gift code created")
        setCreateOpen(false)
        setCode("")
        setProvider("manual")
        setTotalCredits("50")
        setValidUntil("")
        fetchCodes()
      } else {
        toast.error("Failed: " + (await res.text()))
      }
    } catch {
      toast.error("Network error")
    }
  }

  const generateRandomCode = () => {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    let result = "GIFT-"
    for (let i = 0; i < 10; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length))
    }
    setCode(result)
  }

  if (forbidden) {
    return (
      <div className="p-8 max-w-4xl mx-auto">
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-8 text-center">
          <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
          <h2 className="text-lg font-semibold text-foreground">Admin access required</h2>
          <p className="text-sm text-muted-foreground mt-1">You don't have permission to view this page.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Gift className="h-6 w-6 text-[#0052CC]" />
            Gift Codes
          </h1>
          <p className="text-sm text-muted-foreground mt-1">Manage promotional and partner gift codes.</p>
        </div>
        <Button className="bg-[#0052CC] hover:bg-[#003D99] text-white" onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4 mr-1.5" />
          Create Code
        </Button>
      </div>

      {loading ? (
        <div className="text-sm text-muted-foreground py-8 text-center">Loading...</div>
      ) : codes.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border p-12 text-center">
          <Gift className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
          <p className="text-sm text-muted-foreground">No gift codes created yet.</p>
        </div>
      ) : (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-border bg-muted/30">
              <tr className="text-left text-xs text-muted-foreground uppercase">
                <th className="px-4 py-3 font-medium">Code</th>
                <th className="px-4 py-3 font-medium">Provider</th>
                <th className="px-4 py-3 font-medium">Credits</th>
                <th className="px-4 py-3 font-medium">Used</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Valid Until</th>
                <th className="px-4 py-3 font-medium">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {codes.map((c) => {
                const redeemed = c.redeemed_by && c.redeemed_by !== ""
                const expired = new Date(c.valid_until) < new Date()
                return (
                  <tr key={c.code} className="hover:bg-muted/20">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-foreground">{c.code}</span>
                        <button onClick={() => { navigator.clipboard.writeText(c.code); toast.success("Copied") }} className="text-muted-foreground hover:text-foreground">
                          <Copy className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{c.provider}</td>
                    <td className="px-4 py-3 text-foreground">${c.total_credits.toFixed(2)}</td>
                    <td className="px-4 py-3 text-muted-foreground">${(c.used_credits || 0).toFixed(2)}</td>
                    <td className="px-4 py-3">
                      {redeemed ? (
                        <span className="rounded-full bg-[#36B37E]/15 px-2 py-0.5 text-[10px] font-medium text-[#36B37E]">Redeemed</span>
                      ) : expired ? (
                        <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-[10px] font-medium text-red-500">Expired</span>
                      ) : (
                        <span className="rounded-full bg-[#0052CC]/15 px-2 py-0.5 text-[10px] font-medium text-[#0052CC]">Active</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{new Date(c.valid_until).toLocaleDateString()}</td>
                    <td className="px-4 py-3 text-muted-foreground">{new Date(c.created_at).toLocaleDateString()}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="border-border bg-card">
          <DialogHeader>
            <DialogTitle className="text-foreground">Create Gift Code</DialogTitle>
            <DialogDescription>Generate a redeemable gift code for promotional campaigns or partner credits.</DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-4">
            <div>
              <Label className="text-xs text-muted-foreground mb-1.5 block">Code</Label>
              <div className="flex gap-2">
                <Input placeholder="e.g. LAUNCH2026" className="border-border bg-background font-mono" value={code} onChange={(e) => setCode(e.target.value.toUpperCase())} />
                <Button variant="outline" size="sm" onClick={generateRandomCode}>Generate</Button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5 block">Provider</Label>
                <select className="w-full h-9 rounded-md border border-border bg-background px-3 text-sm" value={provider} onChange={(e) => setProvider(e.target.value)}>
                  <option value="manual">Manual</option>
                  <option value="aws-activate">AWS Activate</option>
                  <option value="google-for-startups">Google for Startups</option>
                  <option value="stripe-atlas">Stripe Atlas</option>
                  <option value="nvidia-inception">NVIDIA Inception</option>
                  <option value="partner">Partner</option>
                  <option value="conference">Conference</option>
                </select>
              </div>
              <div>
                <Label className="text-xs text-muted-foreground mb-1.5 block">Credits (USD)</Label>
                <Input type="number" min="1" step="1" className="border-border bg-background" value={totalCredits} onChange={(e) => setTotalCredits(e.target.value)} />
              </div>
            </div>
            <div>
              <Label className="text-xs text-muted-foreground mb-1.5 block">Valid Until (optional, defaults to +1 year)</Label>
              <Input type="date" className="border-border bg-background" value={validUntil} onChange={(e) => setValidUntil(e.target.value)} />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button className="bg-[#0052CC] hover:bg-[#003D99] text-white" onClick={handleCreate}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
