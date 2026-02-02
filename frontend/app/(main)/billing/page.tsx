"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
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
  CreditCard,
  Gift,
  AlertCircle,
  Check,
  Calendar,
} from "lucide-react"

export default function BillingPage() {
  const [currentPlan] = useState({
    name: "Alpha",
    price: 0,
    renewalDate: "Mar 1, 2026",
    features: ["50,000 monthly credits", "Unlimited models", "Priority support", "API access"],
  })

  const [creditBalance] = useState({
    total: 5.00,
    creditId: "sch-7842",
    gifted: 5.00,
    monthly: 0.00,
    purchased: 0.00,
    resetDays: 21,
  })

  const [cancelPlanOpen, setCancelPlanOpen] = useState(false)
  const [redeemCodeOpen, setRedeemCodeOpen] = useState(false)
  const [redeemCode, setRedeemCode] = useState("")

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
                  <span className="text-xl font-bold text-foreground">{currentPlan.name}</span>
                  <span className="text-lg font-semibold text-foreground">${currentPlan.price}<span className="text-xs font-normal text-muted-foreground">/mo</span></span>
                </div>
              </div>
              <span className="rounded-full bg-[#0052CC]/10 px-2 py-0.5 text-xs font-medium text-[#0052CC] dark:text-[#2684FF]">Active</span>
            </div>
            
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-3">
              <Calendar className="h-3 w-3" />
              <span>Renews {currentPlan.renewalDate}</span>
            </div>

            <div className="flex flex-wrap gap-1.5 mb-3">
              {currentPlan.features.map((feature, i) => (
                <span key={i} className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                  <Check className="h-3 w-3 text-green-500" />
                  {feature}
                </span>
              ))}
            </div>

            <div className="flex gap-2">
              <Button variant="outline" size="sm" className="flex-1 bg-transparent text-xs opacity-50 cursor-not-allowed" disabled>
                View Plans (soon)
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Credit Balance Card */}
        <Card className="border-border bg-card">
          <CardContent className="p-4">
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-xs text-muted-foreground">Credit Balance · Resets in <span className="font-medium text-foreground">{creditBalance.resetDays}d</span></p>
              </div>
              <Button size="sm" disabled className="h-7 text-xs bg-[#0052CC]/50 text-white cursor-not-allowed">
                Buy Credits (soon)
              </Button>
            </div>
            
            <div className="flex gap-3">
              <div className="relative w-24 h-16 rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 dark:from-gray-600 dark:to-gray-800 p-2 flex flex-col justify-between shrink-0">
                <div className="flex justify-end">
                  <div className="w-5 h-3 rounded-sm bg-amber-400/80" />
                </div>
                <div>
                  <div className="text-base font-bold text-white">${creditBalance.total.toFixed(2)}</div>
                  <div className="text-[8px] text-gray-400">{creditBalance.creditId}</div>
                </div>
              </div>

              <div className="flex-1 space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Gifted</span>
                  <span className="text-foreground">${creditBalance.gifted.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Monthly</span>
                  <span className="text-foreground">${creditBalance.monthly.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Purchased</span>
                  <span className="text-foreground">${creditBalance.purchased.toFixed(2)}</span>
                </div>
                <div className="flex justify-between font-medium border-t border-border pt-1">
                  <span className="text-foreground">Total</span>
                  <span className="text-foreground">${creditBalance.total.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Usage Code Card */}
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
      </div>

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
