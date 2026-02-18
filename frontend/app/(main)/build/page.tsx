"use client"

import { Suspense } from "react"
import { useSearchParams } from "next/navigation"
import { BuildWizard } from "@/components/build/build-wizard"

function BuildContent() {
  const searchParams = useSearchParams()
  const key = searchParams.get("t") || "default"
  return <BuildWizard key={key} />
}

export default function BuildPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-64 text-muted-foreground">Loading...</div>}>
      <BuildContent />
    </Suspense>
  )
}
