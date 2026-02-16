"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Label } from "@/components/ui/label"
import {
  CreditCard,
  Gift,
  Check,
  Calendar,
  Loader2,
  Receipt,
  Plus,
  MoreHorizontal,
  ExternalLink,
  AlertCircle,
  Trash2,
} from "lucide-react"

interface QuotaData {
  plan: string
  credits_total: number
  credits_used: number
  credits_remaining: number
  models_limit: number
  models_used: number
  queries_daily: number
  queries_used: number
  storage_limit_mb: number
  storage_used_mb: number
  reset_date: string
  days_until_reset: number
  datasets_connected: number
}

export default function BillingPage() {
  const [quota, setQuota] = useState<QuotaData | null>(null)
  const [loading, setLoading] = useState(true)
  const [redeemCode, setRedeemCode] = useState("")
  const [redeemCodeOpen, setRedeemCodeOpen] = useState(false)
  const [viewPlansOpen, setViewPlansOpen] = useState(false)
  const [cancelPlanOpen, setCancelPlanOpen] = useState(false)
  const [buyCreditsOpen, setBuyCreditsOpen] = useState(false)
  const [addPaymentOpen, setAddPaymentOpen] = useState(false)
  const [deletePaymentOpen, setDeletePaymentOpen] = useState(false)
  const [creditAmount, setCreditAmount] = useState("25")
  const [cardNumber, setCardNumber] = useState("")
  const [cardExpiry, setCardExpiry] = useState("")
  const [cardCvc, setCardCvc] = useState("")
  const [paymentMethod, setPaymentMethod] = useState<{ type: string; last4: string; expiry: string; brand: string } | null>(null)

  const handleDeletePayment = () => { setPaymentMethod(null); setDeletePaymentOpen(false) }
  const handleAddPayment = () => {
    if (cardNumber && cardExpiry && cardCvc) {
      setPaymentMethod({ type: "card", last4: cardNumber.slice(-4), expiry: cardExpiry, brand: "Visa" })
      setCardNumber(""); setCardExpiry(""); setCardCvc(""); setAddPaymentOpen(false)
    }
  }

  useEffect(() => {
    fetchQuota()
  }, [])

  const fetchQuota = async () => {
    try {
      const res = await fetch("/api/quota", { credentials: "include" })
      if (res.ok) {
        const data = await res.json()
        setQuota(data)
      }
    } catch (e) {
      console.error("Failed to fetch quota:", e)
    } finally {
      setLoading(false)
    }
  }

  const isUnlimited = quota?.plan === "alpha_unlimited"
  const planName = isUnlimited ? "Alpha (Unlimited)" : "Alpha"
  const creditsTotal = isUnlimited ? quota?.credits_used || 0 : quota?.credits_total || 5
  const creditsRemaining = isUnlimited ? 0 : quota?.credits_remaining || 5
  const resetDate = quota?.reset_date ? new Date(quota.reset_date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : "—"
  const daysUntilReset = quota?.days_until_reset || 0

  const features = isUnlimited
    ? ["Unlimited credits", "Unlimited models", "Unlimited queries", "API access", "10 GB storage"]
    : [`$${quota?.credits_total || 5} monthly credits`, `${quota?.models_limit || 5} AI models`, `${quota?.queries_daily || 10} queries/day`, "API access"]

  if (loading) return <div className="flex items-center justify-center h-64 text-muted-foreground"><Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading...</div>

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#0052CC]/10 dark:bg-[#0052CC]/20">
          <CreditCard className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-foreground">Billing</h1>
          <p className="text-xs text-muted-foreground">Manage subscription and credits</p>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Current Plan Card */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-xs text-muted-foreground">Current Plan</p>
                <div className="flex items-baseline gap-2 mt-0.5">
                  <span className="text-xl font-bold text-foreground">{planName}</span>
                </div>
              </div>
              <span className="rounded-full bg-[#0052CC]/10 px-2 py-0.5 text-xs font-medium text-[#0052CC] dark:text-[#2684FF]">Active</span>
            </div>
            
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-3">
              <Calendar className="h-3 w-3" />
              <span>Renews {resetDate}</span>
            </div>

            <div className="flex flex-wrap gap-1.5 mb-3">
              {features.map((feature, i) => (
                <span key={i} className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                  <Check className="h-3 w-3 text-green-500" />
                  {feature}
                </span>
              ))}
            </div>

            <div className="flex gap-2">
              <Button variant="outline" size="sm" className="flex-1 bg-transparent text-xs" onClick={() => setViewPlansOpen(true)}>
                View Plans
              </Button>
              <Button variant="outline" size="sm" className="bg-transparent text-xs text-red-500 hover:text-red-500 hover:bg-red-500/10" onClick={() => setCancelPlanOpen(true)}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Credit Balance Card */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-xs text-muted-foreground">Credit Balance · Resets in <span className="font-medium text-foreground">{daysUntilReset}d</span></p>
              </div>
              <Button size="sm" onClick={() => setBuyCreditsOpen(true)} className="h-7 text-xs bg-[#0052CC] text-white hover:bg-[#003D99]">
                Buy Credits
              </Button>
            </div>
            
            <div className="flex gap-3">
              <div className="relative w-24 h-16 rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 dark:from-gray-600 dark:to-gray-800 p-2 flex flex-col justify-between shrink-0">
                <div className="flex justify-end">
                  <div className="w-5 h-3 rounded-sm bg-amber-400/80" />
                </div>
                <div>
                  <div className="text-base font-bold text-white">
                    {isUnlimited ? "∞" : `$${creditsRemaining.toFixed(2)}`}
                  </div>
                  <div className="text-[8px] text-gray-400">sch-{quota?.plan?.slice(0,4) || "alpha"}</div>
                </div>
              </div>

              <div className="flex-1 space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Gifted</span>
                  <span className="text-foreground">{isUnlimited ? "∞" : `$${(quota?.credits_total || 5).toFixed(2)}`}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Used</span>
                  <span className="text-foreground">${(quota?.credits_used || 0).toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Purchased</span>
                  <span className="text-foreground">$0.00</span>
                </div>
                <div className="flex justify-between font-medium border-t border-border pt-1">
                  <span className="text-foreground">Remaining</span>
                  <span className="text-foreground">{isUnlimited ? "∞" : `$${creditsRemaining.toFixed(2)}`}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Redeem Code Card */}
        <Card className="border-border bg-card lg:col-span-2">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Gift className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />
              <p className="text-xs text-muted-foreground">Redeem gifted credits</p>
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Enter code"
                value={redeemCode}
                onChange={(e) => setRedeemCode(e.target.value)}
                className="h-8 text-xs border-border bg-background"
              />
              <Button size="sm" onClick={() => setRedeemCodeOpen(true)} disabled={!redeemCode.trim()} className="h-8 text-xs bg-[#0052CC] text-white hover:bg-[#003D99]">
                Redeem
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Payment Method Card */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground mb-2">Payment Method</p>
            {paymentMethod ? (
              <div className="flex items-center justify-between rounded border border-border bg-muted/30 p-2">
                <div className="flex items-center gap-2">
                  <div className="flex h-8 w-10 items-center justify-center rounded bg-white dark:bg-gray-800 border border-border">
                    <span className="text-[10px] font-bold text-blue-600">VISA</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-foreground">{paymentMethod.brand} •••• {paymentMethod.last4}</p>
                    <p className="text-xs text-muted-foreground">Exp {paymentMethod.expiry}</p>
                  </div>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="sm" className="h-7 w-7 p-0"><MoreHorizontal className="h-4 w-4" /></Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem className="gap-2 cursor-pointer text-xs"><ExternalLink className="h-3 w-3" />Manage on Stripe</DropdownMenuItem>
                    <DropdownMenuItem className="gap-2 cursor-pointer text-xs text-red-500 focus:text-red-500" onClick={() => setDeletePaymentOpen(true)}><Trash2 className="h-3 w-3" />Remove</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            ) : (
              <Button variant="outline" size="sm" className="w-full gap-2 bg-transparent text-xs" onClick={() => setAddPaymentOpen(true)}>
                <Plus className="h-3 w-3" />Add Payment Method
              </Button>
            )}
          </CardContent>
        </Card>

        {/* Invoices Card */}
        <Card className="border-border bg-card lg:col-span-2">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Receipt className="h-4 w-4 text-[#0052CC] dark:text-[#2684FF]" />
                <div>
                  <p className="text-sm font-medium text-foreground">Invoices</p>
                  <p className="text-xs text-muted-foreground">Managed on Stripe</p>
                </div>
              </div>
              <Button variant="outline" size="sm" className="gap-2 bg-transparent text-xs">
                <ExternalLink className="h-3 w-3" />Open Stripe
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* View Plans Modal */}
      <Dialog open={viewPlansOpen} onOpenChange={setViewPlansOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[550px]">
          <DialogHeader><DialogTitle className="text-foreground">Available Plans</DialogTitle></DialogHeader>
          <div className="grid gap-3 sm:grid-cols-3">
            {[
              { name: "Starter", price: 0, credits: "5K", features: ["5 models", "Community support"] },
              { name: "Pro", price: 49, credits: "50K", features: ["Unlimited models", "Priority support"], current: planName.includes("Alpha") },
              { name: "Enterprise", price: 199, credits: "Unlimited", features: ["Custom models", "Dedicated support"] },
            ].map((plan) => (
              <div key={plan.name} className={`rounded-lg border p-3 ${plan.current ? "border-[#0052CC] bg-[#0052CC]/5" : "border-border"}`}>
                <h3 className="font-medium text-foreground text-sm">{plan.name}</h3>
                <div className="mt-1"><span className="text-xl font-bold text-foreground">${plan.price}</span><span className="text-xs text-muted-foreground">/mo</span></div>
                <p className="text-xs text-muted-foreground">{plan.credits} credits</p>
                <ul className="mt-2 space-y-1">{plan.features.map((f, i) => (<li key={i} className="flex items-center gap-1 text-xs text-muted-foreground"><Check className="h-3 w-3 text-green-500" />{f}</li>))}</ul>
                <Button className={`w-full mt-3 h-7 text-xs ${plan.current ? "bg-muted text-muted-foreground" : "bg-[#0052CC] text-white hover:bg-[#003D99]"}`} disabled={plan.current}>{plan.current ? "Current" : "Upgrade"}</Button>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>

      {/* Cancel Plan Modal */}
      <Dialog open={cancelPlanOpen} onOpenChange={setCancelPlanOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Cancel Subscription</DialogTitle>
            <DialogDescription className="text-sm">Cancel your subscription? Access ends on {resetDate}.</DialogDescription>
          </DialogHeader>
          <div className="rounded border border-amber-500/20 bg-amber-500/10 p-2 text-xs text-amber-600 dark:text-amber-400 flex items-start gap-2">
            <AlertCircle className="h-3 w-3 mt-0.5 shrink-0" /><span>No refunds for the current billing period.</span>
          </div>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setCancelPlanOpen(false)} className="bg-transparent">Keep</Button>
            <Button variant="destructive" size="sm">Cancel Plan</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Buy Credits Modal */}
      <Dialog open={buyCreditsOpen} onOpenChange={setBuyCreditsOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[350px]">
          <DialogHeader><DialogTitle className="text-foreground">Buy Credits</DialogTitle></DialogHeader>
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2">
              {["10", "25", "50", "100", "250", "500"].map((amount) => (
                <Button key={amount} variant={creditAmount === amount ? "default" : "outline"} size="sm" className={creditAmount === amount ? "bg-[#0052CC] text-white" : "bg-transparent"} onClick={() => setCreditAmount(amount)}>${amount}</Button>
              ))}
            </div>
            <Input type="number" placeholder="Custom amount" value={creditAmount} onChange={(e) => setCreditAmount(e.target.value)} className="h-8 text-sm border-border bg-background" />
          </div>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setBuyCreditsOpen(false)} className="bg-transparent">Cancel</Button>
            <Button size="sm" className="bg-[#0052CC] text-white hover:bg-[#003D99]">Purchase ${creditAmount}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add Payment Modal */}
      <Dialog open={addPaymentOpen} onOpenChange={setAddPaymentOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[350px]">
          <DialogHeader><DialogTitle className="text-foreground">Add Payment Method</DialogTitle></DialogHeader>
          <div className="space-y-3">
            <div><Label className="text-xs">Card Number</Label><Input placeholder="4242 4242 4242 4242" value={cardNumber} onChange={(e) => setCardNumber(e.target.value)} className="h-8 text-sm mt-1 border-border bg-background" /></div>
            <div className="grid grid-cols-2 gap-2">
              <div><Label className="text-xs">Expiry</Label><Input placeholder="MM/YY" value={cardExpiry} onChange={(e) => setCardExpiry(e.target.value)} className="h-8 text-sm mt-1 border-border bg-background" /></div>
              <div><Label className="text-xs">CVC</Label><Input placeholder="123" value={cardCvc} onChange={(e) => setCardCvc(e.target.value)} className="h-8 text-sm mt-1 border-border bg-background" /></div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setAddPaymentOpen(false)} className="bg-transparent">Cancel</Button>
            <Button size="sm" onClick={handleAddPayment} disabled={!cardNumber || !cardExpiry || !cardCvc} className="bg-[#0052CC] text-white hover:bg-[#003D99]">Add Card</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Payment Modal */}
      <Dialog open={deletePaymentOpen} onOpenChange={setDeletePaymentOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[350px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Remove Payment Method</DialogTitle>
            <DialogDescription className="text-sm">Remove this card from your account?</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setDeletePaymentOpen(false)} className="bg-transparent">Cancel</Button>
            <Button variant="destructive" size="sm" onClick={handleDeletePayment}>Remove</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Redeem Code Modal */}
      <Dialog open={redeemCodeOpen} onOpenChange={setRedeemCodeOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[350px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Redeem Code</DialogTitle>
            <DialogDescription className="text-sm">Add credits to your account?</DialogDescription>
          </DialogHeader>
          <div className="rounded border border-border bg-muted/50 p-3 text-center">
            <p className="text-xs text-muted-foreground">Code</p>
            <p className="font-mono font-medium text-foreground">{redeemCode}</p>
          </div>
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => { setRedeemCodeOpen(false); setRedeemCode("") }} className="bg-transparent">Cancel</Button>
            <Button size="sm" onClick={() => { setRedeemCodeOpen(false); setRedeemCode("") }} className="bg-[#0052CC] text-white hover:bg-[#003D99]">Confirm</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
