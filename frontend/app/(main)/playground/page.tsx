"use client"

import { Suspense } from "react"
import { PlaygroundContent } from "@/components/playground/playground-content"

export default function PlaygroundPage() {
  return (
    <Suspense fallback={null}>
      <PlaygroundContent />
    </Suspense>
  )
}
