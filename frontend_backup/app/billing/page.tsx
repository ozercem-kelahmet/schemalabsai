"use client"

import { useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { CreditCard, Download, Calendar, CheckCircle2 } from "lucide-react"

export default function BillingPage() {
  const [plan] = useState({ name: "Pro", status: "active", nextBilling: "Jan 30, 2026" })
  const [paymentMethod] = useState({ brand: "Visa", last4: "4242", expiry: "12/27" })
  const [invoices] = useState([
    { id: "1", description: "Pro Plan - Monthly", date: "Dec 30, 2025", status: "paid" },
    { id: "2", description: "Pro Plan - Monthly", date: "Nov 30, 2025", status: "paid" },
    { id: "3", description: "Pro Plan - Monthly", date: "Oct 30, 2025", status: "paid" },
  ])

  return (
    <Sidebar>
      <div className="flex-1 overflow-auto bg-background">
        <div className="max-w-4xl mx-auto p-8">
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-foreground">Billing</h1>
            <p className="text-sm text-muted-foreground mt-1">Manage your subscription and payment methods</p>
          </div>

          <Card className="mb-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Current Plan</CardTitle>
                  <CardDescription>Your active subscription</CardDescription>
                </div>
                <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30">{plan.status}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-bold">{plan.name}</h3>
                <div className="text-right">
                  <p className="text-sm text-muted-foreground">Next billing date</p>
                  <p className="font-medium">{plan.nextBilling}</p>
                </div>
              </div>
              <div className="mt-4 flex gap-2">
                <Button variant="outline" onClick={() => window.location.href = "/upgrade"}>Change Plan</Button>
                <Button variant="ghost" className="text-destructive">Cancel Subscription</Button>
              </div>
            </CardContent>
          </Card>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Payment Method</CardTitle>
              <CardDescription>Your default payment method</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center gap-4">
                  <div className="p-2 bg-secondary rounded-lg">
                    <CreditCard className="h-6 w-6" />
                  </div>
                  <div>
                    <p className="font-medium">{paymentMethod.brand} •••• {paymentMethod.last4}</p>
                    <p className="text-sm text-muted-foreground">Expires {paymentMethod.expiry}</p>
                  </div>
                </div>
                <Button variant="outline" size="sm">Update</Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Billing History</CardTitle>
              <CardDescription>Your past invoices and payments</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {invoices.map((invoice) => (
                  <div key={invoice.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className="p-2 bg-secondary rounded-lg">
                        <Calendar className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="font-medium">{invoice.description}</p>
                        <p className="text-sm text-muted-foreground">{invoice.date}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="flex items-center gap-1">
                        <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                        <span className="text-xs text-muted-foreground capitalize">{invoice.status}</span>
                      </div>
                      <Button variant="ghost" size="sm"><Download className="h-4 w-4" /></Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Sidebar>
  )
}
