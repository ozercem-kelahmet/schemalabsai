import { Suspense } from "react"
import { PlaygroundContent } from "@/components/playground/playground-content"

export default async function PlaygroundSessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  return (
    <Suspense fallback={null}>
      <PlaygroundContent sessionId={id} />
    </Suspense>
  )
}
