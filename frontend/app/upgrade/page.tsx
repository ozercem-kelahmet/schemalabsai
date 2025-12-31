"use client"

import { useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Check, Sparkles, Zap, Building2, Shield, Headphones, BarChart3, Globe, Lock, Cpu } from "lucide-react"

const plans = [
  { 
    name: "Free", 
    price: "$0",
    description: "For individuals getting started", 
    icon: Zap,
    features: [
      "1,000 API calls/month", 
      "5 GB storage", 
      "3 fine-tuned models", 
      "Community support",
      "Basic analytics",
      "Standard API access"
    ], 
    current: true 
  },
  { 
    name: "Pro", 
    price: "$49",
    period: "/month",
    description: "For professionals and teams", 
    icon: Sparkles,
    features: [
      "50,000 API calls/month", 
      "50 GB storage", 
      "Unlimited models", 
      "Priority email support",
      "Advanced analytics",
      "Full API access",
      "Custom integrations",
      "Team collaboration",
      "Export capabilities"
    ], 
    popular: true 
  },
  { 
    name: "Enterprise", 
    price: "Custom",
    description: "For large organizations", 
    icon: Building2,
    features: [
      "Unlimited API calls", 
      "Unlimited storage", 
      "Dedicated support 24/7",
      "Custom SLA",
      "SSO & SAML",
      "On-premise deployment",
      "Audit logs",
      "Custom contracts",
      "Training & onboarding"
    ] 
  }
]

const features = [
  { icon: Shield, title: "Enterprise Security", desc: "SOC 2 Type II compliant with end-to-end encryption" },
  { icon: Headphones, title: "Dedicated Support", desc: "Get help from our expert team whenever you need" },
  { icon: BarChart3, title: "Advanced Analytics", desc: "Deep insights into model performance and usage" },
  { icon: Globe, title: "Global Infrastructure", desc: "Low latency access from anywhere in the world" },
  { icon: Lock, title: "Data Privacy", desc: "Your data stays yours - never used for training" },
  { icon: Cpu, title: "High Performance", desc: "Optimized inference for production workloads" },
]

const faqs = [
  { q: "Can I change plans anytime?", a: "Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately." },
  { q: "What payment methods do you accept?", a: "We accept all major credit cards, wire transfers for Enterprise, and annual billing options." },
  { q: "Is there a free trial for Pro?", a: "Yes, Pro plan comes with a 14-day free trial. No credit card required to start." },
  { q: "What happens to my data if I downgrade?", a: "Your data is preserved. You'll have read-only access to content exceeding your new plan limits." },
]

export default function UpgradePage() {
  const [loading, setLoading] = useState<string | null>(null)

  const handleUpgrade = (planName: string) => {
    if (planName === "Enterprise") { 
      const email = process.env.NEXT_PUBLIC_CONTACT_EMAIL || "hello@schemalabs.ai"
      window.location.href = "mailto:" + email + "?subject=Enterprise Plan Inquiry"
      return 
    }
    if (planName === "Free") return
    setLoading(planName)
    setTimeout(() => { setLoading(null); alert("Coming soon!") }, 500)
  }

  return (
    <Sidebar>
      <div className="flex-1 overflow-auto bg-background">
        <div className="max-w-6xl mx-auto p-8">
          <div className="text-center mb-10">
            <h1 className="text-3xl font-bold mb-2">Choose Your Plan</h1>
            <p className="text-muted-foreground">Scale your AI capabilities with the right plan for your needs</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
            {plans.map((plan) => (
              <Card key={plan.name} className={plan.popular ? "border-primary" : ""}>
                {plan.popular && (
                  <div className="bg-primary text-primary-foreground text-xs text-center py-1">Most Popular</div>
                )}
                <CardHeader className="text-center">
                  <plan.icon className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                  <CardTitle>{plan.name}</CardTitle>
                  <CardDescription>{plan.description}</CardDescription>
                  <div className="mt-4">
                    <span className="text-3xl font-bold">{plan.price}</span>
                    {plan.period && <span className="text-muted-foreground">{plan.period}</span>}
                  </div>
                </CardHeader>
                <CardContent>
                  <Button 
                    className="w-full mb-4" 
                    variant={plan.popular ? "default" : "outline"}
                    disabled={plan.current || loading === plan.name} 
                    onClick={() => handleUpgrade(plan.name)}
                  >
                    {plan.current ? "Current Plan" : plan.name === "Enterprise" ? "Contact Sales" : "Upgrade"}
                  </Button>
                  <ul className="space-y-2">
                    {plan.features.map((f, i) => (
                      <li key={i} className="flex items-center gap-2 text-sm">
                        <Check className="h-4 w-4 text-green-500 flex-shrink-0" />
                        {f}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="mb-16">
            <h2 className="text-2xl font-bold text-center mb-8">Why teams choose SchemaLabs</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {features.map((f, i) => (
                <div key={i} className="p-6 border rounded-lg">
                  <f.icon className="h-8 w-8 mb-4 text-primary" />
                  <h3 className="font-semibold mb-2">{f.title}</h3>
                  <p className="text-sm text-muted-foreground">{f.desc}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="mb-16">
            <h2 className="text-2xl font-bold text-center mb-8">Frequently Asked Questions</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {faqs.map((faq, i) => (
                <div key={i} className="p-6 border rounded-lg">
                  <h3 className="font-semibold mb-2">{faq.q}</h3>
                  <p className="text-sm text-muted-foreground">{faq.a}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="text-center p-8 border rounded-lg bg-muted/30">
            <h2 className="text-2xl font-bold mb-2">Need a custom solution?</h2>
            <p className="text-muted-foreground mb-4">Let us discuss your specific requirements and build something great together.</p>
            <Button variant="outline" onClick={() => handleUpgrade("Enterprise")}>
              Contact Sales
            </Button>
          </div>
        </div>
      </div>
    </Sidebar>
  )
}
