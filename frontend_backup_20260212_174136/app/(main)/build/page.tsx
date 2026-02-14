import { Suspense } from "react"
import { BuildWizard } from "@/components/build/build-wizard"

export default function BuildPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-64 text-muted-foreground">Loading...</div>}>
      <BuildWizard />
    </Suspense>
  )
}
