import { NextResponse } from "next/server"
import { cookies } from "next/headers"

export async function GET() {
  const session = (await cookies()).get("session")
  if (!session?.value) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  return NextResponse.json({
    usage: {
      apiCalls: { used: 247, limit: 1000 },
      storage: { used: 1.2, limit: 5 },
      models: { used: 2, limit: 3 },
      queries: { used: 45, limit: 100 }
    },
    history: []
  })
}
